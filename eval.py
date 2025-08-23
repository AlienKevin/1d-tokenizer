#!/usr/bin/env python3
"""
Evaluation script to calculate visual metrics for all generated samples.

This script:
1. Finds all .npz files in results/ directory
2. Detects available GPUs and distributes workers across them
3. Runs evaluations in parallel with configurable number of workers
4. Distributes GPUs evenly across workers (round-robin if workers > GPUs)
5. Parses metrics from output: IS, FID, sFID, precision, recall
6. Saves results to results.json

Usage:
    python eval.py
"""

import os
import json
import subprocess
import concurrent.futures
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import multiprocessing
import fcntl
import time


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU indices.
    
    Returns:
        List of GPU indices (e.g., [0, 1, 2, 3])
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            gpu_indices = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            return gpu_indices
        else:
            print("Warning: nvidia-smi failed, falling back to single GPU (0)")
            return [0]
    except Exception as e:
        print(f"Warning: Could not detect GPUs ({e}), falling back to single GPU (0)")
        return [0]


def monitor_gpu_usage() -> str:
    """Get current GPU usage for monitoring."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "GPU monitoring unavailable"
    except:
        return "GPU monitoring unavailable"


def find_npz_files(results_dir: str = "results") -> List[str]:
    """Find all .npz files in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: {results_dir} directory does not exist")
        return []
    
    npz_files = list(results_path.glob("titok_l_32_sampler_*_scheduler_linear_step_32_temp_9.5_cfg_4.5_decay_linear_sm_True.npz"))
    return [str(f) for f in npz_files]


def extract_id_from_filename(filepath: str) -> str:
    """Extract a clean ID from the npz filename."""
    filename = Path(filepath).stem  # Remove .npz extension
    # Use the filename without extension as the ID
    return filename


def append_result_to_file(sample_id: str, metrics: Dict[str, float], output_file: str, max_retries: int = 5):
    """Safely append a single result to the JSONL file with file locking."""
    result_entry = {
        "sample_id": sample_id,
        "timestamp": time.time(),
        "metrics": metrics
    }
    
    for attempt in range(max_retries):
        try:
            # Open file in append mode
            with open(output_file, 'a') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Write the JSON line
                    json.dump(result_entry, f)
                    f.write('\n')
                    f.flush()  # Ensure data is written to disk
                    print(f"[{sample_id}] Result saved to {output_file}")
                    return True
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except Exception as e:
            print(f"[{sample_id}] Attempt {attempt + 1}/{max_retries} failed to write to {output_file}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
            else:
                print(f"[{sample_id}] Failed to write result after {max_retries} attempts")
                return False
    
    return False


def run_evaluation(ref_file: str, sample_file: str, available_gpus: List[int], worker_idx: int = 0, output_file: str = "results.jsonl") -> Tuple[str, Optional[Dict[str, float]], Optional[str]]:
    """
    Run evaluation for a single sample file.
    
    Args:
        ref_file: Path to reference file
        sample_file: Path to sample file to evaluate
        available_gpus: List of available GPU indices
        worker_idx: Index to determine GPU assignment
    
    Returns:
        (sample_id, metrics_dict, error_message)
    """
    # Assign GPU based on worker index (round-robin)
    gpu_id = available_gpus[worker_idx % len(available_gpus)]
    sample_id = extract_id_from_filename(sample_file)
    
    try:
        # Run the evaluation command
        cmd = [
            "python3", 
            "guided-diffusion/evaluations/evaluator.py", 
            ref_file, 
            sample_file
        ]
        
        # Set GPU for this worker with multiple environment variables for better isolation
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Force TensorFlow to see only the specified GPU
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # TensorFlow-specific GPU visibility
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        # Limit TensorFlow to specific GPU
        env["TF_GPU_THREAD_MODE"] = "gpu_private"
        
        print(f"[Worker {worker_idx}] [{sample_id}] Starting evaluation on GPU {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})...")
        
        # Run with separate stdout/stderr to prevent log overlay
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/svl/u/kevin02/1d-tokenizer",
            env=env,  # Make sure to pass the env
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Command failed with return code {result.returncode}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
            print(f"[Worker {worker_idx}] [{sample_id}] ERROR on GPU {gpu_id}: {error_msg}")
            return sample_id, None, error_msg
        
        # Parse metrics from output
        metrics = parse_metrics(result.stdout)
        if metrics is None:
            error_msg = f"Failed to parse metrics from output:\n{result.stdout}"
            print(f"[Worker {worker_idx}] [{sample_id}] ERROR on GPU {gpu_id}: {error_msg}")
            return sample_id, None, error_msg
        
        print(f"[Worker {worker_idx}] [{sample_id}] Completed successfully on GPU {gpu_id}")
        print(f"[Worker {worker_idx}] [{sample_id}] IS: {metrics['IS']:.4f}, FID: {metrics['FID']:.4f}, sFID: {metrics['sFID']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Immediately append result to file
        append_result_to_file(sample_id, metrics, output_file)
        
        return sample_id, metrics, None
        
    except subprocess.TimeoutExpired:
        error_msg = "Evaluation timed out (1 hour)"
        print(f"[Worker {worker_idx}] [{sample_id}] ERROR on GPU {gpu_id}: {error_msg}")
        return sample_id, None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[Worker {worker_idx}] [{sample_id}] ERROR on GPU {gpu_id}: {error_msg}")
        return sample_id, None, error_msg


def parse_metrics(output: str) -> Optional[Dict[str, float]]:
    """
    Parse evaluation metrics from the evaluator output.
    
    Expected output format:
    ...
    Computing evaluations...
    Inception Score: <value>
    FID: <value>
    sFID: <value>
    Precision: <value>
    Recall: <value>
    """
    try:
        # Look for the metrics in the output
        metrics = {}
        
        # Patterns to match each metric
        patterns = {
            'IS': r'Inception Score:\s*([\d\.]+)',
            'FID': r'FID:\s*([\d\.]+)',
            'sFID': r'sFID:\s*([\d\.]+)',
            'precision': r'Precision:\s*([\d\.]+)',
            'recall': r'Recall:\s*([\d\.]+)'
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[metric_name] = float(match.group(1))
            else:
                print(f"Warning: Could not find {metric_name} in output")
                return None
        
        # Ensure we found all required metrics
        required_metrics = ['IS', 'FID', 'sFID', 'precision', 'recall']
        if all(metric in metrics for metric in required_metrics):
            return metrics
        else:
            missing = [m for m in required_metrics if m not in metrics]
            print(f"Warning: Missing metrics: {missing}")
            return None
            
    except Exception as e:
        print(f"Error parsing metrics: {e}")
        return None


def main():
    """Main evaluation function."""
    print("=== Visual Metrics Evaluation ===")
    
    # Configuration
    ref_file = "VIRTUAL_imagenet256_labeled.npz"
    results_dir = "/tmp/kevin02/gcs/tok1d"
    # results_dir = "results"
    output_file = "results.jsonl"
    
    # Worker configuration - can be more than number of GPUs
    # GPUs will be distributed evenly across all workers in round-robin fashion
    max_workers = 4
    
    # Detect available GPUs
    available_gpus = get_available_gpus()
    
    print(f"Detected {len(available_gpus)} GPU(s): {available_gpus}")
    print(f"Using {max_workers} workers with ProcessPoolExecutor for proper GPU isolation")
    print(f"GPU assignment strategy: Round-robin (worker_idx % {len(available_gpus)})")
    
    # Show current GPU status
    print("\nCurrent GPU status:")
    print(monitor_gpu_usage())
    
    if max_workers > len(available_gpus):
        print(f"\nNote: {max_workers} workers will share {len(available_gpus)} GPUs")
        print(f"Expected distribution: ~{max_workers // len(available_gpus)} workers per GPU")
    else:
        print(f"\nEach worker will use a dedicated GPU")
    
    # Check reference file exists
    if not os.path.exists(ref_file):
        print(f"Error: Reference file {ref_file} not found")
        sys.exit(1)
    
    # Find all sample files
    sample_files = find_npz_files(results_dir)
    if not sample_files:
        print(f"No .npz files found in {results_dir} directory")
        sys.exit(1)
    
    print(f"\nFound {len(sample_files)} sample files to evaluate")
    print(f"Reference file: {ref_file}")
    print(f"Sample files: {[os.path.basename(f) for f in sample_files]}")
    print()
    
    # Initialize output file (clear it if it exists)
    if os.path.exists(output_file):
        print(f"Clearing existing {output_file}")
        open(output_file, 'w').close()
    
    # Run evaluations in parallel
    results = {}
    errors = {}
    
    # Use ProcessPoolExecutor for better process isolation and GPU assignment
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs with worker-based GPU assignment
        future_to_file = {}
        for i, sample_file in enumerate(sample_files):
            # Worker index for GPU assignment (round-robin)
            worker_idx = i
            future = executor.submit(run_evaluation, ref_file, sample_file, available_gpus, worker_idx, output_file)
            future_to_file[future] = sample_file
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            sample_file = future_to_file[future]
            try:
                sample_id, metrics, error = future.result()
                
                if metrics is not None:
                    results[sample_id] = metrics
                else:
                    errors[sample_id] = error
                    
            except Exception as e:
                sample_id = extract_id_from_filename(sample_file)
                error_msg = f"Future failed: {str(e)}"
                print(f"[{sample_id}] ERROR: {error_msg}")
                errors[sample_id] = error_msg
    
    # Save results
    print()
    print("=== Final GPU Status ===")
    print(monitor_gpu_usage())
    print()
    print("=== Results Summary ===")
    print(f"Successfully evaluated: {len(results)} files")
    print(f"Failed evaluations: {len(errors)} files")
    
    if errors:
        print("\nErrors:")
        for sample_id, error in errors.items():
            print(f"  {sample_id}: {error}")
    
    if results:
        print(f"\nResults have been written to {output_file} as they completed")
        
        print("\nSuccessful evaluations:")
        for sample_id, metrics in results.items():
            print(f"  {sample_id}:")
            print(f"    IS: {metrics['IS']:.4f}")
            print(f"    FID: {metrics['FID']:.4f}")
            print(f"    sFID: {metrics['sFID']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
    
    print(f"\nEvaluation complete. All results written to {output_file}")
    print(f"Note: Results are in JSONL format (one JSON object per line)")


if __name__ == "__main__":
    main()
