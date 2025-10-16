#!/usr/bin/env python3
"""
PyTorch Profiler Analysis Script

This script helps analyze PyTorch profiling results and provides insights
into performance bottlenecks in your model training.

Usage:
    python analyze_profiler.py --profiler_dir ./profiler/ml-1m-l200/your_model_dir/
"""

import argparse
import os
import torch
from torch.profiler import profile, ProfilerActivity
import json


def analyze_profiler_trace(profiler_dir: str):
    """Analyze profiler trace and print key insights."""
    
    if not os.path.exists(profiler_dir):
        print(f"Profiler directory {profiler_dir} does not exist!")
        return
    
    # Find the latest trace file
    trace_files = [f for f in os.listdir(profiler_dir) if f.endswith('.json')]
    if not trace_files:
        print(f"No trace files found in {profiler_dir}")
        return
    
    latest_trace = max(trace_files, key=lambda x: os.path.getctime(os.path.join(profiler_dir, x)))
    trace_path = os.path.join(profiler_dir, latest_trace)
    
    print(f"Analyzing trace file: {trace_path}")
    
    # Load and analyze the trace
    with open(trace_path, 'r') as f:
        trace_data = json.load(f)
    
    # Extract key metrics
    events = trace_data.get('traceEvents', [])
    
    # Group events by name and calculate statistics
    event_stats = {}
    for event in events:
        if event.get('ph') == 'X':  # Complete events
            name = event.get('name', 'unknown')
            duration = event.get('dur', 0) / 1000.0  # Convert to milliseconds
            
            if name not in event_stats:
                event_stats[name] = []
            event_stats[name].append(duration)
    
    # Calculate statistics for each event type
    print("\n=== Performance Analysis ===")
    print(f"{'Event Name':<40} {'Count':<8} {'Avg (ms)':<12} {'Max (ms)':<12} {'Total (ms)':<12}")
    print("-" * 90)
    
    sorted_events = sorted(event_stats.items(), key=lambda x: sum(x[1]), reverse=True)
    
    for name, durations in sorted_events[:20]:  # Top 20 events
        count = len(durations)
        avg_duration = sum(durations) / count
        max_duration = max(durations)
        total_duration = sum(durations)
        
        print(f"{name:<40} {count:<8} {avg_duration:<12.2f} {max_duration:<12.2f} {total_duration:<12.2f}")
    
    # Identify potential bottlenecks
    print("\n=== Potential Bottlenecks ===")
    high_avg_events = [(name, sum(durations)/len(durations)) for name, durations in event_stats.items() 
                       if sum(durations)/len(durations) > 10]  # Events taking >10ms on average
    
    if high_avg_events:
        for name, avg_time in sorted(high_avg_events, key=lambda x: x[1], reverse=True):
            print(f"- {name}: {avg_time:.2f}ms average")
    else:
        print("No obvious bottlenecks found (all events <10ms average)")
    
    # Memory analysis if available
    memory_events = [e for e in events if 'memory' in e.get('name', '').lower()]
    if memory_events:
        print(f"\n=== Memory Events ===")
        print(f"Found {len(memory_events)} memory-related events")
    
    print(f"\n=== Recommendations ===")
    print("1. Use TensorBoard to visualize the trace:")
    print(f"   tensorboard --logdir {profiler_dir}")
    print("2. Look for:")
    print("   - Long-running operations in forward/backward pass")
    print("   - Memory allocation patterns")
    print("   - GPU utilization gaps")
    print("   - Data loading bottlenecks")


def main():
    parser = argparse.ArgumentParser(description='Analyze PyTorch profiler results')
    parser.add_argument('--profiler_dir', type=str, required=True,
                       help='Directory containing profiler traces')
    
    args = parser.parse_args()
    analyze_profiler_trace(args.profiler_dir)


if __name__ == "__main__":
    main()
