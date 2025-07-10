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
from ray.train.torch import TorchConfig, TorchTrainer

def get_dataloaders(batch_size_per_worker):
    """
    Using your original FakeData generator.
    This is fast and requires no downloads.
    """
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
    """
    This is a simplified and robust training loop based on the FashionMNIST example.
    It uses a standard training pattern without any complex features like mixed precision.
    """
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]

    world_rank = ray.train.get_context().get_world_rank()
    print(f"[Rank {world_rank}] Process started, assigned to GPU {ray.train.get_context().get_local_rank()}.")

    # Get dataloader and prepare it for distributed training
    train_dataloader = get_dataloaders(batch_size=batch_size)
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)

    # Using your original model to match the workload
    model = models.convnext_base(weights=None, num_classes=1000)
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Simple, standard training loop
    for epoch in range(epochs):
        model.train()
        # Required for the distributed sampler to shuffle correctly across epochs
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch} Rank {world_rank}"):
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # We removed the test loop and ray.train.report for maximum simplicity to debug the NCCL error.
        print(f"[Rank {world_rank}] Finished training epoch {epoch}.")
        
    print(f"[Rank {world_rank}] Training finished.")


def run_training_job(num_nodes=2, gpus_per_node=8):
    """
    This is your launcher function, adapted to run the new, simpler training loop.
    """
    total_workers = num_nodes * gpus_per_node
    
    print(f"Total Workers (world_size): {total_workers}")
    print(f"GPUs per Worker:            1")
    print("-------------------------------------------------------------")

    train_config = {
        "lr": 1e-3,  # Using a more standard learning rate
        "epochs": 10,
        "batch_size_per_worker": 64,
    }

    # Configure computation resources
    scaling_config = ScalingConfig(
        num_workers=total_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )
    
    # Standard TorchConfig for NCCL
    torch_config = TorchConfig(backend="nccl")

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
    )

    # Start distributed training
    result = trainer.fit()
    print(f"\n--- Training Run Complete ---")
    print(f"Training result: {result}")


if __name__ == "__main__":
    # This block makes the script self-contained and easy to run.
    run_training_job(num_nodes=2, gpus_per_node=8)
