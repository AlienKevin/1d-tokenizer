#!/usr/bin/env python3
"""
Pareto analysis script for results.jsonl
Creates plots showing best results for different metrics vs parameters (step, cfg, temp)
"""

import json
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse

def parse_sample_id(sample_id):
    """Extract step, temp, cfg, and sampler from sample_id string"""
    # Extract step
    step_match = re.search(r'step_(\d+)', sample_id)
    step = int(step_match.group(1)) if step_match else None
    
    # Extract temp
    temp_match = re.search(r'temp_([\d.]+)', sample_id)
    temp = float(temp_match.group(1)) if temp_match else None
    
    # Extract cfg
    cfg_match = re.search(r'cfg_([\d.]+)', sample_id)
    cfg = float(cfg_match.group(1)) if cfg_match else None
    
    # Extract sampler type (random if _rand_ is present, otherwise confidence)
    sampler = 'random' if '_rand_' in sample_id else 'confidence'
    
    return step, temp, cfg, sampler

def load_data(filename):
    """Load and parse the results.jsonl file"""
    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                
                # Skip entries matching sampler_*_scheduler pattern
                if 'sampler_' in entry['sample_id'] and '_scheduler' in entry['sample_id']:
                    continue
                
                step, temp, cfg, sampler = parse_sample_id(entry['sample_id'])
                
                # Skip entries with missing parameters
                if step is None or temp is None or cfg is None:
                    continue
                
                metrics = entry['metrics']
                data.append({
                    'sample_id': entry['sample_id'],
                    'step': step,
                    'temp': temp,
                    'cfg': cfg,
                    'sampler': sampler,
                    'IS': metrics['IS'],
                    'FID': metrics['FID'],
                    'sFID': metrics['sFID'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid line: {line[:50]}... Error: {e}")
                continue
    
    return pd.DataFrame(data)

def find_best_results(df, group_by, metrics):
    """Find best results for each metric grouped by parameter(s)"""
    results = {}
    
    for metric in metrics:
        # Determine if higher or lower is better
        if metric in ['FID', 'sFID']:
            # Lower is better
            best_results = df.loc[df.groupby(group_by)[metric].idxmin()]
        else:
            # Higher is better (IS, precision, recall)
            best_results = df.loc[df.groupby(group_by)[metric].idxmax()]
        
        # Remove any NaN results
        best_results = best_results.dropna(subset=[metric])
        results[metric] = best_results
    
    return results

def create_plots(df, output_dir='plots'):
    """Create all the requested plots with dual curves for each sampler"""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['FID', 'IS', 'sFID', 'precision', 'recall']
    params = ['step', 'cfg', 'temp']
    samplers = ['random', 'confidence']
    colors = {'random': '#1f77b4', 'confidence': '#ff7f0e'}  # Blue and orange
    markers = {'random': 'o', 'confidence': 's'}  # Circle and square
    
    # Set up the plot style
    plt.style.use('default')
    fig_size = (12, 8)
    
    for param in params:
        # Create subplot figure for all metrics vs this parameter
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Best Results: Metrics vs {param.upper()} (by Sampler)', fontsize=16)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes_flat[i]
            
            has_data = False
            
            for sampler in samplers:
                # Filter data by sampler
                sampler_df = df[df['sampler'] == sampler]
                
                if len(sampler_df) == 0:
                    continue
                
                # Find best results for this metric grouped by parameter
                best_results = find_best_results(sampler_df, param, [metric])[metric]
                
                if len(best_results) == 0:
                    continue
                
                has_data = True
                
                # Sort by parameter for plotting
                best_results = best_results.sort_values(param)
                
                # Plot
                x_vals = best_results[param]
                y_vals = best_results[metric]
                
                ax.plot(x_vals, y_vals, marker=markers[sampler], color=colors[sampler], 
                       linewidth=2, markersize=8, alpha=0.8, label=f'{sampler} sampler')
            
            if not has_data:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
            else:
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            ax.set_xlabel(param.upper())
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs {param.upper()}')
        
        # Remove the extra subplot
        fig.delaxes(axes_flat[5])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_metrics_vs_{param}.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to save memory
        
        # Also create individual plots for each metric
        for metric in metrics:
            plt.figure(figsize=fig_size)
            
            has_data = False
            
            for sampler in samplers:
                # Filter data by sampler
                sampler_df = df[df['sampler'] == sampler]
                
                if len(sampler_df) == 0:
                    continue
                
                best_results = find_best_results(sampler_df, param, [metric])[metric]
                
                if len(best_results) == 0:
                    continue
                
                has_data = True
                best_results = best_results.sort_values(param)
                
                x_vals = best_results[param]
                y_vals = best_results[metric]
                
                plt.plot(x_vals, y_vals, marker=markers[sampler], color=colors[sampler], 
                        linewidth=3, markersize=10, alpha=0.8, label=f'{sampler} sampler')
                
                # Add value annotations
                for x, y in zip(x_vals, y_vals):
                    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                               xytext=(0,15), ha='center', fontsize=9, 
                               color=colors[sampler], fontweight='bold')
            
            if not has_data:
                plt.text(0.5, 0.5, f'No data for {metric}', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
            else:
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.xlabel(param.upper(), fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.title(f'Best {metric} vs {param.upper()} (by Sampler)', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/pareto_{metric}_vs_{param}.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to save memory

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Unique steps: {sorted(df['step'].unique())}")
    print(f"Unique temps: {sorted(df['temp'].unique())}")
    print(f"Unique cfgs: {sorted(df['cfg'].unique())}")
    print(f"Unique samplers: {sorted(df['sampler'].unique())}")
    
    # Print sampler distribution
    print(f"\nSampler distribution:")
    sampler_counts = df['sampler'].value_counts()
    for sampler, count in sampler_counts.items():
        print(f"{sampler:12s}: {count:4d} samples")
    
    print("\nMetric ranges:")
    for metric in ['FID', 'IS', 'sFID', 'precision', 'recall']:
        min_val = df[metric].min()
        max_val = df[metric].max()
        print(f"{metric:10s}: {min_val:8.3f} - {max_val:8.3f}")
    
    print("\nBest overall results:")
    for metric in ['FID', 'IS', 'sFID', 'precision', 'recall']:
        if metric in ['FID', 'sFID']:
            best_idx = df[metric].idxmin()
            best_val = df[metric].min()
            direction = "lowest"
        else:
            best_idx = df[metric].idxmax()
            best_val = df[metric].max()
            direction = "highest"
        
        best_row = df.loc[best_idx]
        print(f"{metric:10s}: {best_val:8.3f} ({direction}) at step={best_row['step']}, temp={best_row['temp']}, cfg={best_row['cfg']}, sampler={best_row['sampler']}")
    
    print("\nBest results by sampler:")
    for sampler in sorted(df['sampler'].unique()):
        print(f"\n{sampler.upper()} SAMPLER:")
        sampler_df = df[df['sampler'] == sampler]
        for metric in ['FID', 'IS', 'sFID', 'precision', 'recall']:
            if metric in ['FID', 'sFID']:
                best_idx = sampler_df[metric].idxmin()
                best_val = sampler_df[metric].min()
                direction = "lowest"
            else:
                best_idx = sampler_df[metric].idxmax()
                best_val = sampler_df[metric].max()
                direction = "highest"
            
            best_row = sampler_df.loc[best_idx]
            print(f"  {metric:10s}: {best_val:8.3f} ({direction}) at step={best_row['step']}, temp={best_row['temp']}, cfg={best_row['cfg']}")

def main():
    parser = argparse.ArgumentParser(description='Create Pareto plots from results.jsonl')
    parser.add_argument('--input', default='results.jsonl', help='Input JSON lines file')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_data(args.input)
    
    if len(df) == 0:
        print("No valid data found!")
        return
    
    print(f"Loaded {len(df)} valid samples")
    
    print_summary(df)
    
    print("\nCreating plots...")
    create_plots(df, args.output_dir)
    
    print(f"\nDone! Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
