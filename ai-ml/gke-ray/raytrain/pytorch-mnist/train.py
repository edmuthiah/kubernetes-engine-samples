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
    """Creates dataloaders using FakeData to simulate a large-scale dataset."""
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
    """The core training function, executed by each of the 16 worker processes."""
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = config["batch_size_per_worker"]

    local_rank = ray.train.get_context().get_local_rank()
    torch.cuda.set_device(local_rank)
    
    world_rank = ray.train.get_context().get_world_rank()
    print(f"[Rank {world_rank}] Process started, assigned to physical GPU {torch.cuda.current_device()}.")
    
    train_dataloader = get_dataloaders(batch_size_per_worker=batch_size_per_worker)
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    
    model = models.convnext_base(weights=None, num_classes=1000)
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch} Rank {world_rank}"):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(X)
                loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # For simplicity in this benchmark, we only report at the end.
        
    print(f"[Rank {world_rank}] Training finished.")


def run_stress_test(num_nodes=2, gpus_per_node=8, cpus_per_node=220):
    """Configures and launches the Ray TorchTrainer job using a one-process-per-GPU model."""
    total_workers = num_nodes * gpus_per_node
    gpus_per_worker = 1
    cpus_per_worker = cpus_per_node // gpus_per_node
    batch_size_per_gpu = 64
    
    print(f"Total Workers (world_size): {total_workers}")
    print(f"GPUs per Worker:            {gpus_per_worker}")
    print("-------------------------------------------------------------")

    train_config = {
        "lr": 0.1,
        "epochs": 10,
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
    print("\n--- Training Run Complete ---")
    print(f"Result: {result}")

if __name__ == "__main__":
    run_stress_test(num_nodes=2, gpus_per_node=8)
