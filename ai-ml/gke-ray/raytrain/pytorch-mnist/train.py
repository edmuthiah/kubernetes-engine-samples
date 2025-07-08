# train.py
#
# A multi-node, multi-GPU stress test for Ray Train using ResNet-50 on the CIFAR-10 dataset.
# This version excludes `torch.compile` for simpler environment compatibility.

import os
from typing import Dict
from filelock import FileLock
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig, TorchCheckpoint

def get_dataloaders(batch_size_per_worker):
    """
    Creates high-performance dataloaders for CIFAR-10 with data augmentation.
    """
    # Data augmentation and normalization for training
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Normalization for validation
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Use a file lock to prevent race conditions on data download across workers
    with FileLock(os.path.expanduser("~/.data.lock")):
        training_data = datasets.CIFAR10(
            root="~/data", train=True, download=True, transform=train_transform
        )
        test_data = datasets.CIFAR10(
            root="~/data", train=False, download=True, transform=test_transform
        )

    # Create DataLoaders with multiple workers and pinned memory for performance
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size_per_worker,
        shuffle=False,  # The distributed sampler handles shuffling
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size_per_worker,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def train_func_per_worker(config: Dict):
    """
    The core training function that is executed on each Ray Train worker.
    """
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = config["batch_size_per_worker"]

    # 1. Get and prepare dataloaders for distributed training
    train_dataloader, test_dataloader = get_dataloaders(
        batch_size_per_worker=batch_size_per_worker
    )
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)
    
    # 2. Get and prepare model for distributed training
    model = models.resnet50(weights=None, num_classes=10)
    model = ray.train.torch.prepare_model(model)

    # NOTE: `torch.compile(model)` has been removed as requested.

    # 3. Define loss function, optimizer, and AMP scaler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Training loop
    for epoch in range(epochs):
        # Set epoch for the distributed sampler to ensure proper shuffling
        if ray.train.get_context().get_world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            # Use autocast for mixed-precision training on Tensor Cores
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(X)
                loss = loss_fn(pred, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # 5. Evaluation loop
        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred = model(X)
                    loss = loss_fn(pred, y)

                test_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()
        
        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total

        # 6. Report metrics and checkpoint using the modern Ray Train API
        # When using DDP, the original model is accessed via `model.module`
        # checkpoint = TorchCheckpoint.from_model(model=model.module)
        
        ray.train.report(
            metrics={"loss": test_loss, "accuracy": accuracy},
            # checkpoint=checkpoint,
        )


def run_stress_test(num_workers=2, gpus_per_worker=8, cpus_per_worker=32):
    """
    Configures and launches the Ray TorchTrainer job.
    """
    # A large batch size per GPU is key to maximizing utilization
    batch_size_per_gpu = 256
    total_gpus = num_workers * gpus_per_worker
    
    # Each Ray worker process manages `gpus_per_worker`, so its total batch size is multiplied.
    batch_size_per_worker = batch_size_per_gpu * gpus_per_worker
    global_batch_size = batch_size_per_worker * num_workers

    print("--- Starting Training Stress Test (non-compiled) ---")
    print(f"Number of nodes (workers): {num_workers}")
    print(f"GPUs per worker:           {gpus_per_worker}")
    print(f"Total GPUs:                {total_gpus}")
    print(f"Batch size per GPU:        {batch_size_per_gpu}")
    print(f"Batch size per worker:       {batch_size_per_worker}")
    print(f"Global batch size:           {global_batch_size}")
    print("--------------------------------------------------")

    train_config = {
        "lr": 0.1,  # A higher learning rate is common for large-batch training
        "epochs": 10,
        "batch_size_per_worker": batch_size_per_worker,
    }

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": gpus_per_worker, "CPU": cpus_per_worker},
    )

    # Use the NCCL backend for fast multi-GPU/multi-node communication
    torch_config = TorchConfig(backend="nccl")

    # Initialize the Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
    )
    
    # Execute the training job
    result = trainer.fit()
    print("\n--- Training Finished ---")
    print(f"Final accuracy: {result.metrics.get('accuracy', 'N/A')}")
    print(f"Training result: {result}")
    print(f"Last checkpoint stored at: {result.checkpoint}")


if __name__ == "__main__":
    # Configure the test for your 2-node, 8-GPU-each setup.
    # `cpus_per_worker` should be high enough to support the DataLoader `num_workers`.
    # 4 data loader workers per GPU is a good rule of thumb, so 4 * 8 = 32.
    run_stress_test(num_workers=2, gpus_per_worker=8, cpus_per_worker=32)
