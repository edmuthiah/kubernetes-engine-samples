import ray
import os
import torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# This dictionary holds the NCCL environment variables we want to inspect.
# We expect these to be set correctly by the wrapper script in the RayCluster YAML.
EXPECTED_ENV_VARS = [
    "LD_LIBRARY_PATH",
    "NCCL_NET",
    "NCCL_NET_PLUGIN",
    "NCCL_IB_HCA",
    "NCCL_DEBUG",
]

def diagnostic_func_per_worker():
    """
    This function runs on each of the 16 worker processes.
    It prints the environment and then tries to initialize torch.distributed.
    """
    world_rank = ray.train.get_context().get_world_rank()
    local_rank = ray.train.get_context().get_local_rank()
    node_id = ray.get_runtime_context().get_node_id()
    
    # --- Part 1: Print the Environment ---
    print(f"\n--- [Rank {world_rank} on Node {node_id}] ---")
    print(f"Assigned to local GPU: {local_rank}")
    
    print("\n[Inspecting Environment Variables]")
    all_vars_ok = True
    for var in EXPECTED_ENV_VARS:
        value = os.environ.get(var, "!!!! NOT SET !!!!")
        print(f"  {var} = {value}")
        if "NOT SET" in value and var != "NCCL_NET": # NCCL_NET is allowed to be unset
            all_vars_ok = False

    if all_vars_ok:
        print("\n[SUCCESS] Critical environment variables appear to be set.")
    else:
        print("\n[FAILURE] Critical environment variables are MISSING. NCCL will likely fail or use the wrong network.")

    # --- Part 2: Attempt NCCL Initialization ---
    # This is the step that fails with "NCCL Error 3" when the environment is wrong.
    try:
        print(f"\n[Rank {world_rank}] Attempting torch.distributed.init_process_group...")
        # The TorchTrainer already initializes the process group for us,
        # but we can check if it's available.
        if torch.distributed.is_initialized():
            backend = torch.distributed.get_backend()
            print(f"[SUCCESS] torch.distributed is initialized! Backend is '{backend}'.")
            # A simple all_reduce to confirm communication
            tensor = torch.ones(1).cuda(local_rank)
            torch.distributed.all_reduce(tensor)
            print(f"[SUCCESS] Rank {world_rank} completed a test AllReduce operation.")
        else:
            # This case shouldn't happen with TorchTrainer, but is a good sanity check
            print("[FAILURE] torch.distributed was NOT initialized by the trainer.")

    except Exception as e:
        print(f"\n[CRITICAL FAILURE] An error occurred during NCCL/torch.distributed setup on Rank {world_rank}: {e}")

    print(f"--- [Rank {world_rank}] Diagnostic complete. ---\n")


def run_test():
    """Sets up and runs the diagnostic test."""
    
    # Use the same scaling as your failing job
    scaling_config = ScalingConfig(
        num_workers=16,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )

    trainer = TorchTrainer(
        train_loop_per_worker=diagnostic_func_per_worker,
        scaling_config=scaling_config,
    )
    
    print("--- Starting diagnostic trainer. Checking environment on 16 workers... ---")
    try:
        result = trainer.fit()
        print("\n--- Diagnostic run finished. See worker logs above for details. ---")
    except Exception as e:
        print(f"\n--- Diagnostic run FAILED. This confirms a fundamental setup issue. Error: {e} ---")

if __name__ == "__main__":
    run_test()
