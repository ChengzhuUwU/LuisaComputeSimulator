#!/usr/bin/env python3
"""
Profiling analysis script for LuisaComputeSimulator
Reads profile JSON files and generates a tree-like performance report
"""

import json
import glob
import os
import sys
from collections import defaultdict
import statistics


def load_profile_files(pattern="profile_*.json"):
    """Load all profile files matching the pattern"""
    files = glob.glob(pattern)
    profiles = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                profiles.append((f, data))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return profiles


def aggregate_profiles(profiles):
    """Aggregate multiple profile runs into average statistics"""
    # Group by frame name
    frame_groups = defaultdict(list)
    for filename, data in profiles:
        frame_name = data.get('name', 'unknown')
        frame_groups[frame_name].append(data)
    
    aggregated = {}
    for frame_name, frames in frame_groups.items():
        aggregated[frame_name] = aggregate_nodes(frames)
    
    return aggregated


def aggregate_nodes(frames):
    """Aggregate node times across multiple frames"""
    if not frames:
        return None
    
    # Collect all unique node paths
    node_times = defaultdict(list)
    
    def collect_times(node, path=""):
        current_path = f"{path}/{node['name']}"
        node_times[current_path].append(node.get('elapsed_ms', 0))
        for child in node.get('children', []):
            collect_times(child, current_path)
    
    for frame in frames:
        collect_times(frame)
    
    # Calculate statistics
    stats = {}
    for path, times in node_times.items():
        stats[path] = {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'count': len(times),
            'total': sum(times)
        }
    
    return stats


def build_average_tree(frames):
    """Build an average tree structure from multiple frames"""
    if not frames:
        return None
    
    def average_node(node_list, name="root"):
        if not node_list:
            return None
        
        # Average elapsed time
        times = [n.get('elapsed_ms', 0) for n in node_list]
        avg_time = statistics.mean(times) if times else 0
        
        result = {
            'name': name,
            'elapsed_ms': avg_time,
            'children': []
        }
        
        # Group children by name
        child_groups = defaultdict(list)
        for node in node_list:
            for child in node.get('children', []):
                child_name = child.get('name', 'unknown')
                child_groups[child_name].append(child)
        
        # Recursively average children
        for child_name, child_list in child_groups.items():
            avg_child = average_node(child_list, child_name)
            if avg_child:
                result['children'].append(avg_child)
        
        return result
    
    return average_node(frames)


def print_tree(node, indent=0, total_time=None, file=sys.stdout):
    """Print a tree structure with timing information"""
    if node is None:
        return
    
    name = node.get('name', 'unknown')
    elapsed = node.get('elapsed_ms', 0)
    
    if total_time is None:
        total_time = elapsed
        percentage = 100.0
    else:
        percentage = (elapsed / total_time * 100) if total_time > 0 else 0
    
    prefix = "  " * indent
    bar = "█" * int(percentage / 5)
    print(f"{prefix}{name}: {elapsed:.3f} ms ({percentage:.1f}%) {bar}", file=file)
    
    for child in node.get('children', []):
        print_tree(child, indent + 1, total_time, file)


def generate_markdown_report(profiles, output_file="profile_report.md"):
    """Generate a markdown report from profile data"""
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("# Performance Profiling Report\n\n")
        f.write(f"Generated from {len(profiles)} profile files\n\n")
        
        # Separate CPU and GPU profiles
        cpu_profiles = [(fn, p) for fn, p in profiles if 'cpu' in fn.lower()]
        gpu_profiles = [(fn, p) for fn, p in profiles if 'gpu' in fn.lower()]
        
        if cpu_profiles:
            f.write("## CPU Profiling Results\n\n")
            f.write("```\n")
            avg_tree = build_average_tree([p for _, p in cpu_profiles])
            print_tree(avg_tree, file=f)
            f.write("```\n\n")
        
        if gpu_profiles:
            f.write("## GPU Profiling Results\n\n")
            f.write("```\n")
            avg_tree = build_average_tree([p for _, p in gpu_profiles])
            print_tree(avg_tree, file=f)
            f.write("```\n\n")
        
        # Detailed statistics
        f.write("## Detailed Statistics\n\n")
        aggregated = aggregate_profiles(profiles)
        
        for frame_name, stats in aggregated.items():
            f.write(f"### {frame_name}\n\n")
            f.write("| Path | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Count |\n")
            f.write("|------|-----------|-------------|----------|----------|-------|\n")
            
            for path, s in sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True):
                f.write(f"| {path} | {s['mean']:.3f} | {s['median']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['count']} |\n")
            
            f.write("\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze profiling data')
    parser.add_argument('--pattern', default='profile_*.json', help='File pattern to match')
    parser.add_argument('--output', default='Document/profile_report.md', help='Output markdown file')
    parser.add_argument('--show', action='store_true', help='Show results in terminal')
    args = parser.parse_args()
    
    profiles = load_profile_files(args.pattern)
    
    if not profiles:
        print(f"No profile files found matching '{args.pattern}'")
        return 1
    
    print(f"Loaded {len(profiles)} profile files")
    
    if args.show:
        cpu_profiles = [(fn, p) for fn, p in profiles if 'cpu' in fn.lower()]
        gpu_profiles = [(fn, p) for fn, p in profiles if 'gpu' in fn.lower()]
        
        if cpu_profiles:
            print("\n=== CPU Profiling Results ===")
            avg_tree = build_average_tree([p for _, p in cpu_profiles])
            print_tree(avg_tree)
        
        if gpu_profiles:
            print("\n=== GPU Profiling Results ===")
            avg_tree = build_average_tree([p for _, p in gpu_profiles])
            print_tree(avg_tree)
    
    generate_markdown_report(profiles, args.output)
    print(f"\nReport saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
