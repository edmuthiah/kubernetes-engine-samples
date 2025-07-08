import os
import time
from typing import Dict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

import ray.train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchConfig

def get_dataloaders(batch_size_per_worker):
    """
    Creates dataloaders using FakeData to simulate a large-scale dataset
    without requiring disk I/O or downloads.
    """
    # Use ImageNet-standard transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = datasets.FakeData(
        size=1_280_000, image_size=(3, 224, 224), num_classes=1000, transform=transform
    )
    test_dataset = datasets.FakeData(
        size=50_000, image_size=(3, 224, 224), num_classes=1000, transform=transform
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_worker,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_worker,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def train_func_per_worker(config: Dict):
    """The core training function that is executed on each Ray Train worker."""
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = config["batch_size_per_worker"]

    # 1. Get and prepare dataloaders for distributed training
    train_dataloader, test_dataloader = get_dataloaders(
        batch_size_per_worker=batch_size_per_worker
    )
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)
    
    # 2. Get and prepare a larger, more modern model
    model = models.convnext_base(weights=None, num_classes=1000)
    model = ray.train.torch.prepare_model(model)

    # 4. Define loss function, optimizer, and AMP scaler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    # 5. Training loop
    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(X)
                loss = loss_fn(pred, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_duration = time.time() - epoch_start_time
        num_images = len(train_dataloader.dataset)
        images_per_second = num_images / epoch_duration
                
        # 6. Report metrics
        ray.train.report(metrics={"images_per_second": images_per_second})


def run_stress_test(num_workers=2, gpus_per_worker=8, cpus_per_worker=32):
    """Configures and launches the Ray TorchTrainer job."""
    batch_size_per_gpu = 256
    batch_size_per_worker = batch_size_per_gpu * gpus_per_worker
    
    print("--- Starting ADVANCED Training Stress Test ---")
    print(f"Model:                     ConvNeXt-Base")
    print(f"Dataset:                   FakeData (ImageNet-sized)")
    print(f"Performance:               torch.compile() enabled")
    print(f"Number of nodes (workers): {num_workers}")
    print(f"GPUs per worker:           {gpus_per_worker}")
    print("-------------------------------------------------")

    train_config = {
        "lr": 0.1,
        "epochs": 5, # Even a few epochs will be very long now
        "batch_size_per_worker": batch_size_per_worker,
    }

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": gpus_per_worker, "CPU": cpus_per_worker},
    )

    # Disable checkpointing for this pure benchmark run
    run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=0))
    torch_config = TorchConfig(backend="nccl")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )
    
    result = trainer.fit()
    print("\n--- Training Finished ---")
    print(f"Final throughput: {result.metrics.get('images_per_second', 'N/A')} images/sec")
    print(f"Training result: {result}")


if __name__ == "__main__":
    run_stress_test(num_workers=2, gpus_per_worker=8, cpus_per_worker=32)
