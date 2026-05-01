"""
Helper functions for experiment naming and organization.

This module provides utilities to generate consistent experiment names
following the convention: {mode}_{framework}_{date}_{time}
"""

import yaml
from datetime import datetime
import os
import argparse


def load_config():
    """Load naming configuration from YAML"""
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/experiment_naming.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_experiment_name(mode, framework, timestamp=None):
    """
    Generate experiment name following the convention.
    
    Args:
        mode: epoch, convergence, epoch_hashring, convergence_hashring
        framework: vanilla, deepspeed, fsdp, lightning, hf_trainer, wandb
        timestamp: optional datetime object (default: now)
    
    Returns:
        String like 'epoch_vanilla_20260419_101309'
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    date_str = timestamp.strftime('%Y%m%d')
    time_str = timestamp.strftime('%H%M%S')
    
    return f"{mode}_{framework}_{date_str}_{time_str}"


def parse_experiment_name(name):
    """
    Parse experiment name back into components.
    
    Handles composite modes like 'epoch_hashring' and 'convergence_hashring'
    which contain underscores.
    """
    parts = name.split('_')
    
    # Handle composite modes (containing underscores)
    # Valid modes: epoch, convergence, epoch_hashring, convergence_hashring
    if len(parts) >= 4:
        # Check for composite modes
        if parts[0] == 'epoch' and parts[1] == 'hashring':
            mode = 'epoch_hashring'
            framework = parts[2]
            date = parts[3]
            time = parts[4] if len(parts) > 4 else None
        elif parts[0] == 'convergence' and parts[1] == 'hashring':
            mode = 'convergence_hashring'
            framework = parts[2]
            date = parts[3]
            time = parts[4] if len(parts) > 4 else None
        else:
            # Simple mode (epoch or convergence)
            mode = parts[0]
            framework = parts[1]
            date = parts[2]
            time = parts[3]
    else:
        # Fallback for unexpected formats
        mode = parts[0] if len(parts) > 0 else 'unknown'
        framework = parts[1] if len(parts) > 1 else 'unknown'
        date = parts[2] if len(parts) > 2 else 'unknown'
        time = parts[3] if len(parts) > 3 else 'unknown'
    
    return {
        'mode': mode,
        'framework': framework,
        'date': date,
        'time': time,
        'full_name': name
    }


def list_experiments_by_mode(mode):
    """List all experiments of a given mode"""
    exp_dir = os.path.join(os.path.dirname(__file__), '..')
    try:
        return [d for d in os.listdir(exp_dir) 
                if os.path.isdir(os.path.join(exp_dir, d)) and d.startswith(mode)]
    except FileNotFoundError:
        return []


def list_all_experiments():
    """List all experiment directories in the experiment_results folder"""
    exp_dir = os.path.join(os.path.dirname(__file__), '..')
    try:
        return [d for d in os.listdir(exp_dir) 
                if os.path.isdir(os.path.join(exp_dir, d)) and not d.startswith('.')]
    except FileNotFoundError:
        return []


def validate_experiment_name(name, config=None):
    """
    Validate that an experiment name follows the convention.
    
    Returns:
        (is_valid, error_message) tuple
    """
    if config is None:
        config = load_config()
    
    parts = name.split('_')
    
    # Check minimum parts
    if len(parts) < 4:
        return False, f"Name must have at least 4 parts separated by '_', got {len(parts)}"
    
    # Parse the name
    parsed = parse_experiment_name(name)
    
    # Validate mode
    if parsed['mode'] not in config.get('valid_modes', []):
        return False, f"Invalid mode: {parsed['mode']}. Valid modes: {config.get('valid_modes', [])}"
    
    # Validate framework
    if parsed['framework'] not in config.get('valid_frameworks', []):
        return False, f"Invalid framework: {parsed['framework']}. Valid frameworks: {config.get('valid_frameworks', [])}"
    
    # Validate date format (YYYYMMDD)
    if len(parsed['date']) != 8 or not parsed['date'].isdigit():
        return False, f"Invalid date format: {parsed['date']}. Expected YYYYMMDD"
    
    # Validate time format (HHMMSS)
    if len(parsed['time']) != 6 or not parsed['time'].isdigit():
        return False, f"Invalid time format: {parsed['time']}. Expected HHMMSS"
    
    return True, "Valid experiment name"


def main():
    """CLI interface for the helper functions"""
    parser = argparse.ArgumentParser(description='Experiment naming helper')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate experiment name')
    gen_parser.add_argument('--mode', required=True, 
                           choices=['epoch', 'convergence', 'epoch_hashring', 'convergence_hashring'],
                           help='Training mode')
    gen_parser.add_argument('--framework', required=True,
                           choices=['vanilla', 'deepspeed', 'fsdp', 'lightning', 'hf_trainer', 'wandb'],
                           help='Training framework')
    gen_parser.add_argument('--timestamp', help='Timestamp in YYYYMMDD_HHMMSS format (default: now)')
    
    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse experiment name')
    parse_parser.add_argument('name', help='Experiment name to parse')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate experiment name')
    validate_parser.add_argument('name', help='Experiment name to validate')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--mode', help='Filter by mode')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        timestamp = None
        if args.timestamp:
            dt = datetime.strptime(args.timestamp, '%Y%m%d_%H%M%S')
            timestamp = dt
        name = generate_experiment_name(args.mode, args.framework, timestamp)
        print(name)
    
    elif args.command == 'parse':
        parsed = parse_experiment_name(args.name)
        print(f"Mode: {parsed['mode']}")
        print(f"Framework: {parsed['framework']}")
        print(f"Date: {parsed['date']}")
        print(f"Time: {parsed['time']}")
    
    elif args.command == 'validate':
        is_valid, msg = validate_experiment_name(args.name)
        if is_valid:
            print(f"VALID: {msg}")
        else:
            print(f"INVALID: {msg}")
    
    elif args.command == 'list':
        if args.mode:
            experiments = list_experiments_by_mode(args.mode)
            print(f"Experiments with mode '{args.mode}':")
        else:
            experiments = list_all_experiments()
            print("All experiments:")
        for exp in sorted(experiments):
            print(f"  {exp}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
