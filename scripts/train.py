#!/usr/bin/env python3
"""
Training script for dual-use mitigation with adversarial unlearning.
"""
import sys
import os
import yaml
import argparse
import torch
from transformers import GPT2Tokenizer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import DualHeadGPT2
from src.data import load_safe_data, load_harmful_data, create_data_loaders
from src.training import AdversarialTrainer
from src.utils import set_seed, get_device, create_experiment_dir


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train dual-use mitigation model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=None,
        help='Specific layer to train (overrides config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Determine which layers to train
    if args.layer is not None:
        layers_to_train = [args.layer]
    else:
        layers_to_train = config['model']['branch_layers']
    
    print(f"Training on layers: {layers_to_train}")
    
    # Load data
    print("Loading data...")
    safe_texts = load_safe_data(
        config['data'].get('safe_data_path'),
        config['data'].get('num_safe_samples')
    )
    harmful_queries = load_harmful_data(
        config['data'].get('harmful_data_path'),
        config['data'].get('num_harmful_samples')
    )
    
    print(f"Loaded {len(safe_texts)} safe samples")
    print(f"Loaded {len(harmful_queries)} harmful samples")
    
    # Create data loaders
    safe_loader, harmful_loader = create_data_loaders(
        safe_texts,
        harmful_queries,
        tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['training']['max_length'],
        shuffle=True,
    )
    
    # Train for each layer
    for branch_layer in layers_to_train:
        print(f"\n{'='*60}")
        print(f"Training model with branch at layer {branch_layer}")
        print(f"{'='*60}\n")
        
        # Create model
        model = DualHeadGPT2(
            model_name=config['model']['name'],
            branch_layer=branch_layer,
            lambda_adv=config['model']['lambda_adv'],
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
        )
        
        # Create trainer
        trainer = AdversarialTrainer(
            model=model,
            safe_loader=safe_loader,
            harmful_loader=harmful_loader,
            optimizer=optimizer,
            device=device,
            lambda_adv=config['model']['lambda_adv'],
            grad_accum_steps=config['training']['grad_accum_steps'],
            max_grad_norm=config['training']['max_grad_norm'],
        )
        
        # Create experiment directory
        exp_dir = create_experiment_dir(
            config['paths']['experiment_dir'],
            branch_layer
        )
        
        # Train
        trainer.train(
            num_epochs=config['training']['num_epochs'],
            save_dir=exp_dir,
            save_interval=config['logging']['save_interval'],
            log_interval=config['logging']['log_interval'],
        )
        
        print(f"\nCompleted training for layer {branch_layer}")
        print(f"Results saved to: {exp_dir}\n")


if __name__ == '__main__':
    main()

