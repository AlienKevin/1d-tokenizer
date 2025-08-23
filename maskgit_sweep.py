#!/usr/bin/env python3
"""
Parameter sweep script for TiTok sampling hyperparameters.

This script creates temporary config files and runs torchrun for each
combination of sampling hyperparameters.
"""

import os
import subprocess
import shutil
import itertools
import tempfile
import yaml
from pathlib import Path
import torch


def create_temp_config(base_config_path, num_steps, randomize_temperature, guidance_scale, guidance_decay, softmax_temperature_annealing, scheduler_mode, sampler_type):
    """Create a temporary config file with modified hyperparameters."""
    
    # Read the base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify the hyperparameters
    config['model']['generator']['num_steps'] = num_steps
    config['model']['generator']['randomize_temperature'] = randomize_temperature
    config['model']['generator']['guidance_scale'] = guidance_scale
    config['model']['generator']['guidance_decay'] = guidance_decay
    config['model']['generator']['scheduler_mode'] = scheduler_mode
    config['model']['generator']['sampler_type'] = sampler_type
    
    # Create temporary config file
    temp_dir = Path("temp_configs ")
    temp_dir.mkdir(exist_ok=True)
    
    temp_config_path = temp_dir / f"titok_l32_sampler_{sampler_type}_scheduler_{scheduler_mode}_steps_{num_steps}_temp_{randomize_temperature}_cfg_{guidance_scale}.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(temp_config_path)


def run_torchrun(config_path, num_steps, randomize_temperature, guidance_scale, guidance_decay, softmax_temperature_annealing, scheduler_mode, sampler_type, sample_tokens_only=False):
    """Run the torchrun command with the specified config and output directory."""
    
    # Detect available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"🔍 Detected {num_gpus} available GPU(s)")
    else:
        num_gpus = 1
        print("⚠️  No CUDA GPUs detected, using CPU (nproc_per_node=1)")
    
    output_dir = f"{out_dir}/titok_l_32_sampler_{sampler_type}_scheduler_{scheduler_mode}_step_{num_steps}_temp_{randomize_temperature}_cfg_{guidance_scale}_decay_{guidance_decay}_sm_{softmax_temperature_annealing}"
    
    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={num_gpus}",
        "--rdzv-endpoint=localhost:9919",
        "sample_imagenet_titok.py",
        f"config={config_path}",
        f"experiment.output_dir={output_dir}"
    ]
    
    if sample_tokens_only:
        cmd.append("experiment.sample_tokens_only=true")
    
    print(f"\n{'='*80}")
    print(f"Running experiment with:")
    print(f"  num_steps: {num_steps}")
    print(f"  randomize_temperature: {randomize_temperature}")
    print(f"  guidance_scale: {guidance_scale}")
    print(f"  guidance_decay: {guidance_decay}")
    print(f"  softmax_temperature_annealing: {softmax_temperature_annealing}")
    print(f"  scheduler_mode: {scheduler_mode}")
    print(f"  sampler_type: {sampler_type}")
    print(f"  num_gpus: {num_gpus}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Successfully completed experiment: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed experiment: {output_dir}")
        print(f"Error: {e}")
        return False


def main():
    """Main function to run the parameter sweep."""
    
    # Check for --sample-tokens-only flag
    import sys
    sample_tokens_only = '--sample-tokens-only' in sys.argv
    if sample_tokens_only:
        print("🔍 Running in sample-tokens-only mode (skipping image decoding)")
    
    # Define hyperparameter ranges
    # num_steps_values = [8, 16, 32]
    # randomize_temperature_values = [9.0, 0.0, 3.0, 6.0, 12.0]
    # guidance_scale_values = [4.5, 0, 1.5, 3.0, 6.0, 7.5, 9.0, 10.5]
    num_steps_values = [32]
    randomize_temperature_values = [9.5]
    guidance_scale_values = [4.5]
    scheduler_mode_values = ['linear']
    # sampler_type_values = ['confidence', 'random', 'fixed']
    sampler_type_values = ['confidence']
    
    # Fixed parameters
    guidance_decay = "linear"
    softmax_temperature_annealing = True
    
    # Base config path
    base_config_path = "configs/infer/TiTok/titok_l32.yaml"
    
    if not os.path.exists(base_config_path):
        print(f"❌ Base config file not found: {base_config_path}")
        return
    
    # Create results directory if it doesn't exist
    global out_dir
    # out_dir = "/tmp/kevin02/gcs/tok1d"
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate all combinations
    total_combinations = len(num_steps_values) * len(randomize_temperature_values) * len(guidance_scale_values) * len(scheduler_mode_values) * len(sampler_type_values)
    print(f"🚀 Starting parameter sweep with {total_combinations} combinations")
    print(f"Parameters:")
    print(f"  num_steps: {num_steps_values}")
    print(f"  randomize_temperature: {randomize_temperature_values}")
    print(f"  guidance_scale: {guidance_scale_values}")
    print(f"  scheduler_mode: {scheduler_mode_values}")
    print(f"  sampler_type: {sampler_type_values}")
    print(f"  guidance_decay: {guidance_decay} (fixed)")
    print(f"  softmax_temperature_annealing: {softmax_temperature_annealing} (fixed)")
    
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0
    
    for i, (num_steps, randomize_temperature, guidance_scale, scheduler_mode, sampler_type) in enumerate(
        itertools.product(num_steps_values, randomize_temperature_values, guidance_scale_values, scheduler_mode_values, sampler_type_values), 1
    ):
        print(f"\n📊 Progress: {i}/{total_combinations}")
        
        # Create temporary config
        temp_config_path = create_temp_config(
            base_config_path,
            num_steps,
            randomize_temperature,
            guidance_scale,
            guidance_decay,
            softmax_temperature_annealing,
            scheduler_mode,
            sampler_type
        )
        
        # Check if this experiment already exists before running
        output_path = f"{out_dir}/titok_l_32_sampler_{sampler_type}_scheduler_{scheduler_mode}_step_{num_steps}_temp_{randomize_temperature}_cfg_{guidance_scale}_decay_{guidance_decay}_sm_{softmax_temperature_annealing}.npz"
        if os.path.exists(output_path):
            skipped_runs += 1
            print(f"⏭️  Skipping experiment: already exists")
            continue
            
        try:
            # Run the experiment
            success = run_torchrun(
                temp_config_path,
                num_steps,
                randomize_temperature,
                guidance_scale,
                guidance_decay,
                softmax_temperature_annealing,
                scheduler_mode,
                sampler_type,
                sample_tokens_only
            )
            
            if success:
                successful_runs += 1
            else:
                failed_runs += 1
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            failed_runs += 1

    print(f"\n🎉 Parameter sweep completed!")
    print(f"✅ Successful runs: {successful_runs}")
    print(f"⏭️  Skipped runs: {skipped_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"📊 Total experiments: {total_combinations}")


if __name__ == "__main__":
    main()
