#!/usr/bin/env python3
"""
Bandwidth Plot Generator for CSE 554 Assignment 1
Generates bandwidth vs transfer size plot from log.log file
"""

import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log_bandwidth(log_file):
    """Parse bandwidth data from log file (first run only)"""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    data = {
        'pageable': {'sizes': [], 'h2d': [], 'd2h': []},
        'pinned': {'sizes': [], 'h2d': [], 'd2h': []}
    }

    current_section = None
    first_run_complete = False

    for line in lines:
        # Stop after first run
        if 'PEAK BANDWIDTH MEASUREMENT' in line and current_section:
            first_run_complete = True

        if first_run_complete:
            break

        # Detect section headers
        if 'Q1: Regular (Pageable) Memory' in line:
            current_section = 'pageable'
        elif 'Q2: Pinned (Page-Locked) Memory' in line:
            current_section = 'pinned'

        # Parse bandwidth lines
        match = re.search(
            r'Size:\s+(\d+)\s+bytes\s+\((pageable|pinned)\)\s+\|\s+H2D:\s+([\d.]+)\s+MB/s\s+\|\s+D2H:\s+([\d.]+)\s+MB/s',
            line
        )
        if match and current_section:
            size = int(match.group(1))
            h2d = float(match.group(3))  # Already in GB/s despite "MB/s" label
            d2h = float(match.group(4))

            data[current_section]['sizes'].append(size)
            data[current_section]['h2d'].append(h2d)
            data[current_section]['d2h'].append(d2h)

    return data


def create_bandwidth_plot(data, output_png='bandwidth_plot_combined.png',
                          output_pdf='bandwidth_plot_combined.pdf'):
    """Create bandwidth vs transfer size plot"""

    # Convert to numpy arrays
    pageable_sizes = np.array(data['pageable']['sizes'])
    pageable_h2d = np.array(data['pageable']['h2d'])
    pageable_d2h = np.array(data['pageable']['d2h'])
    pinned_sizes = np.array(data['pinned']['sizes'])
    pinned_h2d = np.array(data['pinned']['h2d'])
    pinned_d2h = np.array(data['pinned']['d2h'])

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot 4 data curves
    ax.plot(pageable_sizes, pageable_h2d,
            marker='o', linestyle='-', linewidth=2.5, markersize=5,
            label='Pageable Host→Device', color='#1f77b4', alpha=0.8)
    ax.plot(pageable_sizes, pageable_d2h,
            marker='s', linestyle='-', linewidth=2.5, markersize=5,
            label='Pageable Device→Host', color='#ff7f0e', alpha=0.8)
    ax.plot(pinned_sizes, pinned_h2d,
            marker='^', linestyle='--', linewidth=2.5, markersize=6,
            label='Pinned Host→Device', color='#2ca02c', alpha=0.8)
    ax.plot(pinned_sizes, pinned_d2h,
            marker='v', linestyle='--', linewidth=2.5, markersize=6,
            label='Pinned Device→Host', color='#d62728', alpha=0.8)

    # Set x-axis to log scale (base 2)
    ax.set_xscale('log', base=2)

    # Format x-axis labels
    xticks = [2**i for i in range(0, 29, 3)]
    xtick_labels = []
    for i in range(0, 29, 3):
        size = 2**i
        if i <= 10:
            xtick_labels.append(f'$2^{{{i}}}$\n{size}B')
        elif i <= 20:
            xtick_labels.append(f'$2^{{{i}}}$\n{size//1024}KB')
        else:
            xtick_labels.append(f'$2^{{{i}}}$\n{size//(1024**2)}MB')

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=0, ha='center', fontsize=9)

    # Add grid
    ax.grid(True, which='major', linestyle='-', alpha=0.2, linewidth=0.8)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1, linewidth=0.5)

    # Labels and title
    ax.set_xlabel('Transfer Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=13, fontweight='bold')
    ax.set_title('Host-GPU Memory Transfer Bandwidth Comparison\nQuadro RTX 6000 (PCIe Gen3 x16)',
                 fontsize=15, fontweight='bold', pad=20)

    # Set y-axis limit
    ax.set_ylim(-0.2, 14)

    # Add reference line (5th line)
    practical_pcie = 12.5  # GB/s
    ax.axhline(y=practical_pcie, color='darkgray', linestyle='-.', linewidth=2,
               alpha=0.5, zorder=1, label='Practical PCIe Peak (12.5 GB/s)')

    # Add legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
              edgecolor='black', fancybox=True, shadow=True)

    # Add peak annotations
    pinned_h2d_peak = pinned_h2d.max()
    pinned_d2h_peak = pinned_d2h.max()
    pinned_h2d_peak_size = pinned_sizes[pinned_h2d.argmax()]
    pinned_d2h_peak_size = pinned_sizes[pinned_d2h.argmax()]

    if pinned_h2d_peak > 0.1:
        ax.annotate(f'Peak: {pinned_h2d_peak:.2f} GB/s',
                    xy=(pinned_h2d_peak_size, pinned_h2d_peak),
                    xytext=(pinned_h2d_peak_size / 8, pinned_h2d_peak + 1),
                    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
                    fontsize=10, color='#2ca02c', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='#2ca02c', alpha=0.9, linewidth=1.5))

    if pinned_d2h_peak > 0.1:
        ax.annotate(f'Peak: {pinned_d2h_peak:.2f} GB/s',
                    xy=(pinned_d2h_peak_size, pinned_d2h_peak),
                    xytext=(pinned_d2h_peak_size / 8, pinned_d2h_peak - 1.2),
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                    fontsize=10, color='#d62728', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                             edgecolor='#d62728', alpha=0.9, linewidth=1.5))

    plt.tight_layout()

    # Save
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    return pageable_sizes, pageable_h2d, pageable_d2h, pinned_sizes, pinned_h2d, pinned_d2h


def print_statistics(pageable_sizes, pageable_h2d, pageable_d2h,
                    pinned_sizes, pinned_h2d, pinned_d2h):
    """Print bandwidth statistics"""
    print("\n" + "="*80)
    print("BANDWIDTH STATISTICS")
    print("="*80)

    print("\nPageable Memory:")
    print(f"  H2D Peak: {pageable_h2d.max():.3f} GB/s at {pageable_sizes[pageable_h2d.argmax()]:,} bytes")
    print(f"  D2H Peak: {pageable_d2h.max():.3f} GB/s at {pageable_sizes[pageable_d2h.argmax()]:,} bytes")

    print("\nPinned Memory:")
    print(f"  H2D Peak: {pinned_h2d.max():.3f} GB/s at {pinned_sizes[pinned_h2d.argmax()]:,} bytes")
    print(f"  D2H Peak: {pinned_d2h.max():.3f} GB/s at {pinned_sizes[pinned_d2h.argmax()]:,} bytes")

    print("\nImprovement (Pinned vs Pageable):")
    if pageable_h2d.max() > 0:
        print(f"  H2D: {pinned_h2d.max() / pageable_h2d.max():.2f}× faster")
    if pageable_d2h.max() > 0:
        print(f"  D2H: {pinned_d2h.max() / pageable_d2h.max():.2f}× faster")

    print("\nPCIe Efficiency (Pinned Memory vs 12.5 GB/s practical):")
    print(f"  H2D: {(pinned_h2d.max() / 12.5) * 100:.1f}%")
    print(f"  D2H: {(pinned_d2h.max() / 12.5) * 100:.1f}%")

    print("\n" + "="*80)
    print(f"Total lines plotted: 5 (4 data curves + 1 reference line)")
    print("="*80)


def main():
    """Main function"""
    import sys
    import os

    # Get log file path
    log_file = 'log.log' if len(sys.argv) < 2 else sys.argv[1]

    if not os.path.exists(log_file):
        print(f"Error: Log file '{log_file}' not found!")
        print(f"Usage: {sys.argv[0]} [log_file]")
        sys.exit(1)

    print(f"Parsing log file: {log_file}")

    # Parse data
    data = parse_log_bandwidth(log_file)

    # Create plot
    print("Creating bandwidth plot...")
    results = create_bandwidth_plot(data)

    print("✓ Plot saved: bandwidth_plot_combined.png (4 data curves + 1 reference line)")
    print("✓ Plot saved: bandwidth_plot_combined.pdf")

    # Print statistics
    print_statistics(*results)


if __name__ == '__main__':
    main()
