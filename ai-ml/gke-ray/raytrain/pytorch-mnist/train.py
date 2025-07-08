# Copyright 2024 Google LLC
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
from torchvision import datasets, models, transforms

import ray.train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer, TorchConfig

# [MODIFIED] Using a more complex dataset (CIFAR-10) and better DataLoaders
def get_dataloaders(batch_size_per_worker):
    """
    Creates high-performance dataloaders for CIFAR-10.
    """
    # [NEW] Add data augmentation to make the task more challenging
    # and prevent overfitting.
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Use a file lock to prevent multiple workers from downloading the data simultaneously.
    with FileLock(os.path.expanduser("~/.data.lock")):
        training_data = datasets.CIFAR10(
            root="~/data", train=True, download=True, transform=train_transform
        )
        test_data = datasets.CIFAR10(
            root="~/data", train=False, download=True, transform=test_transform
        )

    # [NEW] Use multiple dataloader workers and pin memory for faster data transfer to GPU.
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size_per_worker,
        shuffle=False,  # The distributed sampler will handle shuffling
        num_workers=4, # Increase this based on your CPU cores
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

# [MODIFIED] The training function is significantly updated for performance.
def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size_per_worker = config["batch_size_per_worker"]

    # Get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(
        batch_size_per_worker=batch_size_per_worker
    )
    
    # [1] Prepare Dataloader for distributed training
    train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)
    
    # [MODIFIED] Use a large, standard model like ResNet-50
    model = models.resnet50(weights=None, num_classes=10)

    # [2] Prepare model for distributed training
    model = ray.train.torch.prepare_model(model)
    
    # # This JIT-compiles the model into optimized kernels.
    # print("Compiling the model... (this may take a minute)")
    # model = torch.compile(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # [NEW] Use Automatic Mixed Precision (AMP) for H100/H200 Tensor Cores
    # GradScaler is used to prevent underflow of gradients in float16.
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        if ray.train.get_context().get_world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)

        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            # [NEW] Autocast context manager for mixed precision
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(X)
                loss = loss_fn(pred, y)

            optimizer.zero_grad()
            # [NEW] Scale the loss and call backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        # [3] Report metrics to Ray Train
        ray.train.report(
            metrics={"loss": test_loss, "accuracy": accuracy},
            checkpoint=Checkpoint.from_dict(
                dict(epoch=epoch, model=model.module.state_dict())
            ),
        )

# [MODIFIED] Main function with realistic parameters
def train_cifar_stress_test(num_workers=2, gpus_per_worker=8, cpus_per_worker=16):
    # [NEW] Define batch size on a per-GPU basis for clarity and scalability.
    # H200s can handle large batch sizes. This is a good starting point.
    batch_size_per_gpu = 256

    # Total number of GPUs in the cluster
    total_gpus = num_workers * gpus_per_worker
    
    # [MODIFIED] Calculate a large global batch size to saturate the GPUs
    global_batch_size = batch_size_per_gpu * total_gpus
    
    # This will be the batch size loaded by each of the `num_workers`
    batch_size_per_worker = global_batch_size // num_workers

    print(f"--- Starting Training ---")
    print(f"Number of workers: {num_workers}")
    print(f"GPUs per worker:   {gpus_per_worker}")
    print(f"Total GPUs:        {total_gpus}")
    print(f"Batch size per GPU:  {batch_size_per_gpu}")
    print(f"Batch size per worker: {batch_size_per_worker}")
    print(f"Global batch size:   {global_batch_size}")
    print(f"-------------------------")

    train_config = {
        "lr": 0.1,  # A higher learning rate is often used for large-batch training
        "epochs": 10,
        "batch_size_per_worker": batch_size_per_worker,
    }

    # Your ScalingConfig is already well-suited for your hardware.
    # It assigns 1 worker to each of your 2 nodes.
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": gpus_per_worker, "CPU": cpus_per_worker},
    )

    # Use the NCCL backend for NVIDIA GPU communication
    torch_config = TorchConfig(backend="nccl")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
    )
    
    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == "__main__":
    # Assuming 2 nodes (workers), each with 8 GPUs.
    # Adjust `cpus_per_worker` based on your machine's specs.
    train_cifar_stress_test(num_workers=2, gpus_per_worker=8, cpus_per_worker=16)
