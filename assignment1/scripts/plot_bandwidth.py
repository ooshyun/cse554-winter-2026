#!/usr/bin/env python3
"""
CSE 554 Assignment 1 - Bandwidth Plotting Script
Generate bandwidth vs transfer size plots from CSV data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
csv_path = os.path.join(project_dir, 'srcs', 'cuda', 'host_gpu', 'bandwidth_data.csv')
df = pd.read_csv(csv_path)

# Separate pageable and pinned data
pageable = df[df['memory_type'] == 'pageable']
pinned = df[df['memory_type'] == 'pinned']

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Host-to-Device
ax1.semilogx(pageable['size_bytes'], pageable['h2d_gbps'],
             'o-', label='Pageable', linewidth=2, markersize=4)
ax1.semilogx(pinned['size_bytes'], pinned['h2d_gbps'],
             's-', label='Pinned', linewidth=2, markersize=4)
ax1.set_xlabel('Transfer Size (bytes)', fontsize=12)
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax1.set_title('Host-to-Device Bandwidth', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=1)

# Plot 2: Device-to-Host
ax2.semilogx(pageable['size_bytes'], pageable['d2h_gbps'],
             'o-', label='Pageable', linewidth=2, markersize=4)
ax2.semilogx(pinned['size_bytes'], pinned['d2h_gbps'],
             's-', label='Pinned', linewidth=2, markersize=4)
ax2.set_xlabel('Transfer Size (bytes)', fontsize=12)
ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax2.set_title('Device-to-Host Bandwidth', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=1)

plt.suptitle('CSE 554 Assignment 1 - Memory Transfer Bandwidth Comparison',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

# Save figure
plot_path1 = os.path.join(project_dir, 'profiling_results', 'bandwidth_plot.png')
plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {plot_path1}")

# Also create a combined plot
fig2, ax = plt.subplots(figsize=(10, 7))

ax.semilogx(pageable['size_bytes'], pageable['h2d_gbps'],
            'o-', label='Pageable H2D', linewidth=2, markersize=4, color='#1f77b4')
ax.semilogx(pinned['size_bytes'], pinned['h2d_gbps'],
            's-', label='Pinned H2D', linewidth=2, markersize=4, color='#ff7f0e')
ax.semilogx(pageable['size_bytes'], pageable['d2h_gbps'],
            'o--', label='Pageable D2H', linewidth=2, markersize=4, color='#1f77b4', alpha=0.6)
ax.semilogx(pinned['size_bytes'], pinned['d2h_gbps'],
            's--', label='Pinned D2H', linewidth=2, markersize=4, color='#ff7f0e', alpha=0.6)

ax.set_xlabel('Transfer Size (bytes)', fontsize=13)
ax.set_ylabel('Bandwidth (GB/s)', fontsize=13)
ax.set_title('Memory Transfer Bandwidth: Pageable vs Pinned',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=1)

plt.tight_layout()
plot_path2 = os.path.join(project_dir, 'profiling_results', 'bandwidth_plot_combined.png')
plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
print(f"✓ Combined plot saved to: {plot_path2}")

# Print statistics
print("\n" + "="*70)
print("BANDWIDTH STATISTICS")
print("="*70)

print("\nPageable Memory:")
print(f"  H2D Peak: {pageable['h2d_gbps'].max():.2f} GB/s at {pageable.loc[pageable['h2d_gbps'].idxmax(), 'size_bytes']:,} bytes")
print(f"  D2H Peak: {pageable['d2h_gbps'].max():.2f} GB/s at {pageable.loc[pageable['d2h_gbps'].idxmax(), 'size_bytes']:,} bytes")

print("\nPinned Memory:")
print(f"  H2D Peak: {pinned['h2d_gbps'].max():.2f} GB/s at {pinned.loc[pinned['h2d_gbps'].idxmax(), 'size_bytes']:,} bytes")
print(f"  D2H Peak: {pinned['d2h_gbps'].max():.2f} GB/s at {pinned.loc[pinned['d2h_gbps'].idxmax(), 'size_bytes']:,} bytes")

print("\nSpeedup (Pinned vs Pageable) at 1MB:")
pageable_1mb_h2d = pageable[pageable['size_bytes'] == 1048576]['h2d_gbps'].values[0]
pinned_1mb_h2d = pinned[pinned['size_bytes'] == 1048576]['h2d_gbps'].values[0]
pageable_1mb_d2h = pageable[pageable['size_bytes'] == 1048576]['d2h_gbps'].values[0]
pinned_1mb_d2h = pinned[pinned['size_bytes'] == 1048576]['d2h_gbps'].values[0]

print(f"  H2D: {pinned_1mb_h2d / pageable_1mb_h2d:.2f}x")
print(f"  D2H: {pinned_1mb_d2h / pageable_1mb_d2h:.2f}x")
print("="*70)
