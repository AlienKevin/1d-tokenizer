#!/usr/bin/env python3
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_results(filename):
    """Parse JSONL file and extract data for plotting."""
    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            entry = json.loads(line)
            sample_id = entry['sample_id']
            metrics = entry['metrics']
            
            # Extract CFG value from sample_id
            cfg_match = re.search(r'cfg_(\d+(?:\.\d+)?)', sample_id)
            if not cfg_match:
                continue
            
            cfg_value = float(cfg_match.group(1))
            
            # Determine if this is rand or confidence (no rand)
            is_rand = 'rand' in sample_id and sample_id.find('rand') < sample_id.find('step')
            method = 'rand' if is_rand else 'confidence'
            
            # Add to data
            data.append({
                'cfg': cfg_value,
                'method': method,
                'IS': metrics['IS'],
                'FID': metrics['FID'],
                'sFID': metrics['sFID'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })
    
    return pd.DataFrame(data)

def plot_comparison(df, output_filename='comparison_plot.png'):
    """Create comparison plots for all metrics."""
    
    # Define metrics to plot
    metrics = ['IS', 'FID', 'sFID', 'precision', 'recall']
    
    # Create figure with subplots arranged horizontally
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    
    # Colors for different methods
    colors = {'rand': '#ff7f0e', 'confidence': '#1f77b4'}
    markers = {'rand': 'o', 'confidence': 's'}
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Group data by method
        for method in ['rand', 'confidence']:
            method_data = df[df['method'] == method].copy()
            
            # Handle duplicate CFG values by averaging
            method_data = method_data.groupby('cfg')[metric].mean().reset_index()
            
            # Sort by CFG value
            method_data = method_data.sort_values('cfg')
            
            # Plot
            ax.plot(method_data['cfg'], method_data[metric], 
                   color=colors[method], marker=markers[method], 
                   linewidth=2, markersize=6, label=method.capitalize())
        
        # Formatting
        ax.set_xlabel('CFG Value')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs CFG')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # For FID and sFID, lower is better - consider using log scale if needed
        if metric in ['FID', 'sFID']:
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_data_summary(df):
    """Print a summary of the data."""
    print("Data Summary:")
    print("=" * 50)
    print(f"Total entries: {len(df)}")
    print(f"Methods: {df['method'].unique()}")
    print(f"CFG values: {sorted(df['cfg'].unique())}")
    print("\nEntries per method:")
    print(df['method'].value_counts())
    print("\nEntries per CFG value:")
    print(df['cfg'].value_counts().sort_index())
    print("\n" + "=" * 50)

def main():
    # Parse the data
    filename = 'results_step_32_temp_9.0.jsonl'
    df = parse_results(filename)
    
    if df.empty:
        print(f"No data found in {filename}")
        return
    
    # Print summary
    print_data_summary(df)
    
    # Create the comparison plot
    print("Creating comparison plot...")
    fig = plot_comparison(df)
    
    print("Plot saved as 'comparison_plot.png'")
    
    # Also save the processed data as CSV for reference
    df.to_csv('processed_results.csv', index=False)
    print("Processed data saved as 'processed_results.csv'")

if __name__ == "__main__":
    main()
