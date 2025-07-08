import os
import time
from typing import Dict
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig

def get_dataloaders(batch_size_per_worker):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_dataset = datasets.FakeData(
        size=1_280_000, image_size=(3, 224, 224), num_classes=1000, transform=transform
    )
    return DataLoader(
        train_dataset, batch_size=batch_size_per_worker, shuffle=False, num_workers=4, pin_memory=True
    )

def train_func_per_worker(config: Dict):
    """The core training function that is executed on each Ray Train worker."""
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = config["batch_size_per_worker"]

    # --- DIAGNOSTICS: Print the environment for this specific worker ---
    world_rank = ray.train.get_context().get_world_rank()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
    
    print(
        f"[Rank {world_rank}] Starting. "
        f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}. "
        f"Num CUDA devices seen by torch: {torch.cuda.device_count()}"
    )
    # --- END DIAGNOSTICS ---
    
    # The dataloader uses the per-worker (i.e., per-GPU) batch size
    train_dataloader = get_dataloaders(batch_size_per_worker=batch_size_per_worker)
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    
    model = models.convnext_base(weights=None, num_classes=1000)
    
    # This call prepares the model for DDP and moves it to the correct device
    # based on the worker's assigned GPU.
    model = ray.train.torch.prepare_model(model)

    print(f"[Rank {world_rank}] Model prepared. Current torch device: {torch.cuda.current_device()}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch} Rank {world_rank}"):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(X)
                loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        epoch_duration = time.time() - epoch_start_time
        num_images = len(train_dataloader.dataset)
        images_per_second_this_rank = num_images / epoch_duration
        
        # Report metrics
        ray.train.report(metrics={"images_per_second_per_rank": images_per_second_this_rank})


def run_stress_test(num_nodes=2, gpus_per_node=8, cpus_per_node=32):
    """Configures and launches the Ray TorchTrainer job using a one-process-per-GPU model."""
    total_workers = num_nodes * gpus_per_node
    gpus_per_worker = 1
    cpus_per_worker = cpus_per_node // gpus_per_node
    batch_size_per_gpu = 64
    
    print("--- Starting ADVANCED Training Stress Test (One-Process-Per-GPU) ---")
    print(f"Total Workers (world_size):{total_workers}")
    print(f"GPUs per Worker:           {gpus_per_worker}")
    print("------------------------------------------------------------------")

    train_config = {
        "lr": 0.1,
        "epochs": 20,
        "batch_size_per_worker": batch_size_per_gpu,
    }

    scaling_config = ScalingConfig(
        num_workers=total_workers,
        use_gpu=True,
        resources_per_worker={"GPU": gpus_per_worker, "CPU": cpus_per_worker},
    )
    
    torch_config = TorchConfig(backend="nccl")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
    )
    
    result = trainer.fit()
    print("\n--- Training Finished ---")

if __name__ == "__main__":
    run_stress_test(num_nodes=2, gpus_per_node=8, cpus_per_node=220)
