"""Sampling scripts for TiTok on ImageNet.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: 
    https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
"""

import demo_util
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
import os
import math
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def main():
    config = demo_util.get_config_cli()
    num_fid_samples = 50000
    per_proc_batch_size = 125
    sample_folder_dir = config.experiment.output_dir
    seed = 42

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)

    # setup DDP.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size() 
    device = rank % torch.cuda.device_count()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.") 

    if rank == 0:
        # downloads from hf
        hf_hub_download(repo_id="fun-research/TiTok", filename=f"{config.experiment.tokenizer_checkpoint}", local_dir="./")
        hf_hub_download(repo_id="fun-research/TiTok", filename=f"{config.experiment.generator_checkpoint}", local_dir="./")
    dist.barrier()

    titok_tokenizer = demo_util.get_titok_tokenizer(config)
    titok_generator = demo_util.get_titok_generator(config)
    titok_tokenizer.to(device)
    titok_generator.to(device)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    assert num_fid_samples % global_batch_size == 0
    if rank == 0:
        print(f"Total number of images that will be sampled: {num_fid_samples}")

    samples_needed_this_gpu = int(num_fid_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar, desc='sampling tokens') if rank == 0 else pbar
    total = 0

    all_classes = list(range(config.model.generator.condition_num_classes)) * (num_fid_samples // config.model.generator.condition_num_classes)
    subset_len = len(all_classes) // world_size
    all_classes = np.array(all_classes[rank * subset_len: (rank+1)*subset_len], dtype=np.int64)
    cur_idx = 0

    all_tokens = []

    for _ in pbar:
        y = torch.from_numpy(all_classes[cur_idx * n: (cur_idx+1)*n]).to(device)
        cur_idx += 1

        generated_tokens = demo_util.sample_tokens(
            generator=titok_generator,
            labels=y.long(),
            randomize_temperature=config.model.generator.randomize_temperature,
            softmax_temperature_annealing=True,
            num_sample_steps=config.model.generator.num_steps,
            guidance_scale=config.model.generator.guidance_scale,
            guidance_decay=config.model.generator.guidance_decay,
            device=device
        )

        all_tokens.append(generated_tokens)

    pbar = range(iterations)
    pbar = tqdm(pbar, desc='decoding images') if rank == 0 else pbar

    all_samples = []

    for i in pbar:
        generated_tokens = all_tokens[i]
        samples = demo_util.decode_tokens(generated_tokens=generated_tokens, tokenizer=titok_tokenizer)
        all_samples.append(samples)
        total += global_batch_size

    # Concatenate all samples from this rank
    rank_samples = np.concatenate(all_samples, axis=0)
    print(f"Rank {rank} samples shape: {rank_samples.shape}")

    # Create Gloo group for CPU tensor gathering (works even if default backend is NCCL)
    gloo_group = dist.new_group(backend="gloo")
    
    # Convert to tensor for distributed gathering
    rank_samples_tensor = torch.from_numpy(rank_samples)
    
    # Gather samples from all ranks
    world_size = dist.get_world_size(group=gloo_group)
    gathered_samples = [torch.empty_like(rank_samples_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_samples, rank_samples_tensor, group=gloo_group)
    
    # Create npz directly from gathered samples
    if rank == 0:
        all_gathered_samples = torch.cat(gathered_samples, dim=0).numpy()
        print(f"Total gathered samples shape: {all_gathered_samples.shape}")
        
        npz_path = f"{sample_folder_dir}.npz"
        np.savez(npz_path, arr_0=all_gathered_samples)
        print(f"Saved .npz file to {npz_path} [shape={all_gathered_samples.shape}].")
        print("Done.")
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()