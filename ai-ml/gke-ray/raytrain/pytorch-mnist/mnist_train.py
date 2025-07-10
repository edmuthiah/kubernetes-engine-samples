# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict
from filelock import FileLock
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Normalize, ToTensor

import ray.train
from ray.train import ScalingConfig
# Correct import path for your Ray version
from ray.train.torch import TorchConfig, TorchTrainer


def get_dataloaders(batch_size):
    """
    Downloads the FashionMNIST dataset.
    FileLock is used to prevent multiple processes from downloading the same
    data simultaneously.
    """
    # Transform to normalize the input images
    transform = transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    # We use a file lock to ensure that only one process downloads the data.
    with FileLock(os.path.expanduser("~/data.lock")):
        training_data = datasets.FashionMNIST(
            root="~/data", train=True, download=True, transform=transform
        )
        test_data = datasets.FashionMNIST(
            root="~/data", train=False, download=True, transform=transform
        )

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


class NeuralNetwork(nn.Module):
    """A simple neural network for image classification."""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_func_per_worker(config: Dict):
    """
    This is the robust training loop from the example script.
    It runs on each distributed worker.
    """
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]
    
    world_rank = ray.train.get_context().get_world_rank()
    print(f"[Rank {world_rank}] Starting training worker.")

    # Get dataloaders and prepare them for distributed training.
    train_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size)
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

    # Prepare model for distributed training.
    model = NeuralNetwork()
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Main training and validation loop
    for epoch in range(epochs):
        # Required for the distributed sampler to shuffle correctly.
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch} Rank {world_rank}"):
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item()
                num_total += y.shape[0]
                num_correct += (pred.argmax(1) == y).sum().item()

        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total
        
        # Report metrics back to Ray Train.
        ray.train.report(metrics={"loss": test_loss, "accuracy": accuracy})


def run_fashion_mnist_job(num_nodes=2, gpus_per_node=8):
    """
    This launcher function is adapted to be self-contained like your original script.
    """
    total_workers = num_nodes * gpus_per_node
    global_batch_size = 64  # Total batch size across all workers.

    print(f"Starting FashionMNIST training with {total_workers} workers.")
    print("-------------------------------------------------------------")

    train_config = {
        "lr": 1e-3,
        "epochs": 10,
        # Ensure each worker has at least a batch size of 1.
        "batch_size_per_worker": max(1, global_batch_size // total_workers),
    }

    # Configure computation resources.
    scaling_config = ScalingConfig(
        num_workers=total_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1},
    )

    # Configure the Torch backend.
    torch_config = TorchConfig(backend="nccl")

    # Initialize and run the Ray TorchTrainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
    )
    
    result = trainer.fit()
    print("\n--- Training Run Complete ---")
    print(f"Final result: {result}")


if __name__ == "__main__":
    # This block makes the script self-contained and easy to run.
    run_fashion_mnist_job(num_nodes=2, gpus_per_node=8)
