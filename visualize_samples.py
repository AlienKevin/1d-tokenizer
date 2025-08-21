#!/usr/bin/env python3
"""
Script to create a 3x3 grid of random image samples from a folder.
Usage: python concat_samples.py <folder_path>
"""

import os
import sys
import random
from PIL import Image
import argparse


def get_image_files(folder_path):
    """Get all image files from the specified folder."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    image_files = []
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(folder_path, filename))
    
    return sorted(image_files)


def create_grid(image_paths, grid_size=(3, 3)):
    """Create a grid of images from the given image paths."""
    if len(image_paths) < grid_size[0] * grid_size[1]:
        raise ValueError(f"Need at least {grid_size[0] * grid_size[1]} images, but only {len(image_paths)} found")
    
    # Load and resize images
    images = []
    target_size = None
    
    for path in image_paths:
        try:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Use the first image to determine target size
            if target_size is None:
                target_size = img.size
            else:
                # Resize to match the first image
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue
    
    if len(images) < grid_size[0] * grid_size[1]:
        raise ValueError(f"Could not load enough valid images")
    
    # Create the grid
    grid_width = target_size[0] * grid_size[1]
    grid_height = target_size[1] * grid_size[0]
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    for i, img in enumerate(images[:grid_size[0] * grid_size[1]]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        x = col * target_size[0]
        y = row * target_size[1]
        grid_image.paste(img, (x, y))
    
    return grid_image


def main():
    parser = argparse.ArgumentParser(description='Create a 3x3 grid of random image samples from a folder')
    parser.add_argument('folder_path', help='Path to the folder containing images')
    parser.add_argument('--output', '-o', help='Output filename (default: folder_name.png)')
    parser.add_argument('--seed', type=int, default=3, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Get all image files
    image_files = get_image_files(folder_path)
    
    if len(image_files) < 9:
        print(f"Error: Need at least 9 images, but only {len(image_files)} found in {folder_path}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    # Randomly select 9 images
    selected_images = random.sample(image_files, 9)
    print(f"Selected 9 random images:")
    for img_path in selected_images:
        print(f"  - {os.path.basename(img_path)}")
    
    try:
        # Create the grid
        grid = create_grid(selected_images)
        
        # Determine output filename
        if args.output:
            output_filename = args.output
        else:
            folder_name = os.path.basename(os.path.abspath(folder_path))
            output_filename = f"{folder_name}.png"
        
        # Save the grid
        grid.save(output_filename)
        print(f"Grid saved as: {output_filename}")
        
    except Exception as e:
        print(f"Error creating grid: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


