"""Demo file for sampling images from TiTok.

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
"""


import torch
import io

from omegaconf import OmegaConf
from modeling.titok import TiTok
from modeling.tatitok import TATiTok
from modeling.maskgit import ImageBert, UViTBert
from modeling.rar import RAR
from modeling.maskgen import MaskGen_VQ, MaskGen_KL


def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf

def get_titok_tokenizer(config):
    tokenizer = TiTok(config)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"))
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_tatitok_tokenizer(config):
    tokenizer = TATiTok(config)
    tokenizer.load_state_dict(torch.load(config.experiment.tokenizer_checkpoint, map_location="cpu"))
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

def get_titok_generator(config):
    if config.model.generator.model_type == "ViT":
        model_cls = ImageBert
    elif config.model.generator.model_type == "UViT":
        model_cls = UViTBert
    else:
        raise ValueError(f"Unsupported model type {config.model.generator.model_type}")
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator

def get_rar_generator(config):
    model_cls = RAR
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    generator.set_random_ratio(0)
    return generator

def get_maskgen_vq_generator(config):
    model_cls = MaskGen_VQ
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator

def get_maskgen_kl_generator(config):
    model_cls = MaskGen_KL
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator


@torch.no_grad()
def sample_tokens(generator,
                 labels=None,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=2.0,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8,
                 device="cuda"):
    generator.eval()
    if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps)

    return generated_tokens


@torch.no_grad()
def decode_tokens(generated_tokens, tokenizer):
    tokenizer.eval()
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)

    return generated_image


@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=8,
              device="cuda",
              return_tensor=False):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps)
    
    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image


def is_gcs_path(path):
    """Check if the given path is a Google Cloud Storage URL."""
    return path.startswith("gs://")


def parse_gcs_path(gcs_path):
    """Parse a GCS path into bucket name and blob name.
    
    Args:
        gcs_path (str): GCS path in format gs://bucket/path/to/file
        
    Returns:
        tuple: (bucket_name, blob_name)
    """
    if not is_gcs_path(gcs_path):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    path_without_prefix = gcs_path[5:]  # Remove 'gs://'
    parts = path_without_prefix.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    
    return bucket_name, blob_name


def upload_to_gcs(local_file_path, gcs_path):
    """Upload a local file to Google Cloud Storage.
    
    Args:
        local_file_path (str): Path to the local file
        gcs_path (str): GCS destination path in format gs://bucket/path/to/file
    """
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage is required for GCS support. Install with: pip install google-cloud-storage")
    
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    # Initialize the client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Upload the file
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to {gcs_path}")


def save_npz_to_gcs(data_array, gcs_path):
    """Save a numpy array as .npz file directly to Google Cloud Storage.
    
    Args:
        data_array (numpy.ndarray): Array to save
        gcs_path (str): GCS destination path in format gs://bucket/path/to/file.npz
    """
    try:
        from google.cloud import storage
        import numpy as np
    except ImportError:
        raise ImportError("google-cloud-storage is required for GCS support. Install with: pip install google-cloud-storage")
    
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    # Create in-memory buffer
    buffer = io.BytesIO()
    np.savez(buffer, arr_0=data_array)
    buffer.seek(0)
    
    # Initialize the client and upload
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    blob.upload_from_file(buffer, content_type='application/octet-stream')
    print(f"NPZ file uploaded to {gcs_path} [shape={data_array.shape}]")


def save_samples(samples, output_path):
    """Save samples to either local filesystem or GCS.
    
    Args:
        samples (numpy.ndarray): Array of samples to save
        output_path (str): Output path (local or gs://)
    """
    import numpy as np
    
    if is_gcs_path(output_path):
        # Ensure .npz extension for GCS path
        if not output_path.endswith('.npz'):
            output_path = f"{output_path}.npz"
        save_npz_to_gcs(samples, output_path)
    else:
        # Local filesystem
        if not output_path.endswith('.npz'):
            output_path = f"{output_path}.npz"
        np.savez(output_path, arr_0=samples)
        print(f"Saved .npz file to {output_path} [shape={samples.shape}]")
