import os
import ray
import time
import cupy as cp
import logging
import sys
from ray.util.collective.types import Backend
import ray.util.collective as collective

GROUP = "p2p_benchmark_group"

def setup_logger():
    """Configures a logger to write to stdout and a file in /bucket."""
    try:
        # Get the unique job ID from the Ray runtime context
        job_id = ray.get_runtime_context().get_job_id()
        log_file_path = f"/bucket/{job_id}.log"
    except Exception as e:
        # Fallback if unable to get job ID or if not in a Ray job
        job_id = f"local-run-{int(time.time())}"
        log_file_path = f"/bucket/{job_id}.log"
        print(f"Could not determine Ray job ID, using fallback: {job_id}. Error: {e}", file=sys.stderr)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter for the logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Handler for streaming logs to standard output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Handler for writing logs to the specified file
    try:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails (e.g., /bucket not writable), log an error to stdout
        logging.error(f"CRITICAL: Failed to configure file logging at {log_file_path}. Error: {e}")


@ray.remote(num_gpus=1)
class P2PWorker:
    def __init__(self, world_size: int, rank: int, tensor_gb: int):
        self.rank = rank
        self.world_size = world_size
        self.tensor_size_bytes = int(tensor_gb * 1024 * 1024 * 1024)

        for key, value in os.environ.items():
          print(f"{key}: {value}")

        collective.init_collective_group(
            world_size=self.world_size,
            rank=self.rank,
            backend=Backend.NCCL,
            group_name=GROUP,
        )

    def ready(self):
        """Simple method to check if the actor is initialized."""
        logging.info(f"[Rank {self.rank}] Ready.")
        return True

    def run_benchmark(self, num_iterations=10, direction="send_first"):
        """Runs a timed benchmark for a given direction."""
        tensor = cp.ones(self.tensor_size_bytes, dtype=cp.uint8)
        
        # Determine sender and receiver
        sender_rank, receiver_rank = (0, 1) if direction == "send_first" else (1, 0)
        
        # Warm-up
        for _ in range(5):
            if self.rank == sender_rank:
                collective.send(tensor, dst_rank=receiver_rank, group_name=GROUP)
            elif self.rank == receiver_rank:
                collective.recv(tensor, src_rank=sender_rank, group_name=GROUP)
        cp.cuda.Stream.null.synchronize()
        
        # Timed benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            if self.rank == sender_rank:
                collective.send(tensor, dst_rank=receiver_rank, group_name=GROUP)
            elif self.rank == receiver_rank:
                collective.recv(tensor, src_rank=sender_rank, group_name=GROUP)
        cp.cuda.Stream.null.synchronize()
        duration = time.time() - start_time
        
        # The sender calculates and returns the bandwidth
        if self.rank == sender_rank:
            avg_time = duration / num_iterations
            # Bandwidth in bits per second (for networking Gbps)
            bandwidth_gbps = (self.tensor_size_bytes * 8) / (avg_time * 1e9)
            # Bandwidth in bytes per second (for storage GB/s)
            bandwidth_gbs = self.tensor_size_bytes / (avg_time * 1e9)
            return (bandwidth_gbps, bandwidth_gbs)
        
        return None

    def shutdown(self):
        collective.destroy_collective_group(GROUP)

def main():
    ray.init(address="auto")
    
    # Configure logging for both stdout and a file in /bucket
    setup_logger()

    # --- Default Parameters (formerly argparse) ---
    NUM_NODES = 2
    ITERATIONS = 10
    TENSOR_GB = 4

    if NUM_NODES != 2:
        logging.error("This script is designed for exactly two nodes. Exiting.")
        return

    workers = [
        P2PWorker.options(num_gpus=1).remote(NUM_NODES, rank=i, tensor_gb=TENSOR_GB)
        for i in range(NUM_NODES)
    ]

    # 1. Ensure all workers are initialized before starting
    ray.get([w.ready.remote() for w in workers])
    logging.info("All workers initialized. Starting benchmark...")

    # 2. Run benchmark in direction 0 -> 1
    results_0_to_1 = ray.get([w.run_benchmark.remote(ITERATIONS, "send_first") for w in workers])
    bw_0_to_1 = next(res for res in results_0_to_1 if res is not None)

    # 3. Run benchmark in direction 1 -> 0
    results_1_to_0 = ray.get([w.run_benchmark.remote(ITERATIONS, "recv_first") for w in workers])
    bw_1_to_0 = next(res for res in results_1_to_0 if res is not None)
    
    # --- Format and Log Results ---
    avg_gbps = (bw_0_to_1[0] + bw_1_to_0[0]) / 2
    avg_gbs = (bw_0_to_1[1] + bw_1_to_0[1]) / 2
    
    results_string = (
        f"\n{'='*50}"
        f"\n NCCL Bi-Directional Point-to-Point Benchmark"
        f"\n{'='*50}"
        f"\n  Message Size: {TENSOR_GB} GB"
        f"\n  Iterations: {ITERATIONS}"
        f"\n" + "-" * 50 +
        f"\n  Bandwidth (0 -> 1): {bw_0_to_1[0]:.2f} Gbps ({bw_0_to_1[1]:.2f} GB/s)"
        f"\n  Bandwidth (1 -> 0): {bw_1_to_0[0]:.2f} Gbps ({bw_1_to_0[1]:.2f} GB/s)"
        f"\n" + "-" * 50 +
        f"\n  Average Bandwidth: {avg_gbps:.2f} Gbps ({avg_gbs:.2f} GB/s)"
        f"\n{'='*50}\n"
    )
    logging.info(results_string)

    # --- Shutdown ---
    ray.get([w.shutdown.remote() for w in workers])
    logging.info("Benchmark finished and workers shut down.")

if __name__ == "__main__":
    main()
