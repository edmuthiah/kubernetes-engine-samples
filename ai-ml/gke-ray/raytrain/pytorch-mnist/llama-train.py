import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import ray.train
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train.torch.fsdp import FSDPStrategy

def train_func_per_worker(config: dict):
    """
    The training function for fine-tuning a Llama-3-8B model with FSDP.
    """
    # --- 1. Configuration ---
    set_seed(42)
    model_name = config["model_name"]
    batch_size = config["batch_size_per_worker"]
    lr = config["lr"]
    epochs = config["epochs"]
    
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please set the HUGGING_FACE_TOKEN "
            "environment variable."
        )

    # --- 2. Prepare Model and Tokenizer ---
    model_kwargs = {
        "device_map": "meta", 
        "use_auth_token": hf_token,
        "torch_dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Prepare the Model with FSDP ---
    model = ray.train.torch.prepare_model(model)

    # --- 4. Prepare Dataset ---
    # A simple, synthetic dataset for demonstration purposes.
    # In a real scenario, you would load a dataset from Hugging Face Hub.
    data = ["Ray Train is great for distributed training.", "FSDP helps train giant models."] * 1024
    tokenized_data = tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Create a simple TensorDataset
    dataset = torch.utils.data.TensorDataset(
        tokenized_data["input_ids"], tokenized_data["attention_mask"]
    )
    
    # Create a DataLoader. The DistributedSampler is automatically handled by `prepare_data_loader`.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = ray.train.torch.prepare_data_loader(dataloader)
    
    # --- 5. Training Setup ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        
        # Required for the distributed sampler to shuffle properly across epochs.
        if ray.train.get_context().get_world_rank() == 0:
            print(f"--- Epoch {epoch+1}/{epochs} ---")
            
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            optimizer.step()
            
            if batch_idx % 10 == 0:
                # To report loss, we need to gather it from all workers.
                gathered_loss = ray.train.torch.gather_object(loss.item())
                if ray.train.get_context().get_world_rank() == 0:
                    avg_loss = sum(gathered_loss) / len(gathered_loss)
                    ray.train.report(metrics={"loss": avg_loss})

        # --- 6. Checkpointing ---
        # A more robust checkpointing strategy
        if ray.train.get_context().get_world_rank() == 0:
            print(f"Epoch {epoch+1} finished. Saving checkpoint...")

        checkpoint = Checkpoint.from_dict(
            dict(
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            )
        )
        ray.train.report(metrics={}, checkpoint=checkpoint)


def main():
    model_name = "meta-llama/Meta-Llama-3-8B" 
    num_workers = 2  # One worker per node
    gpus_per_worker = 8 # Each worker uses all 8 GPUs on its node
    batch_size_per_gpu = 4
    
    train_config = {
        "model_name": model_name,
        "epochs": 2,
        "lr": 5e-5,
        "batch_size_per_worker": batch_size_per_gpu,
    }

    fsdp_strategy = FSDPStrategy(
        sharding_strategy="FULL_SHARD", # Shard params, grads, and optimizer states
        auto_wrap_policy="TRANSFORMER_BASED", # Automatically wrap transformer blocks
        activation_checkpointing=True, # Trade compute for memory to fit even larger models
        limit_all_gathers=True, # Optimization for communication
    )

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": gpus_per_worker},
    )

    torch_config = TorchConfig(
        backend="nccl",
        fsdp_strategy=fsdp_strategy
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
    )

    result = trainer.fit()
    print(f"Training result: {result}")
    if result.checkpoint:
        print(f"Last checkpoint saved to: {result.checkpoint.path}")


if __name__ == "__main__":
    main()
