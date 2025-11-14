"""
Training loop with granular gradient steps for adversarial unlearning.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from typing import Dict, Optional

from model import DualHeadGPT2
from utils import save_checkpoint, get_device


class AdversarialTrainer:
    """
    Trainer for adversarial unlearning with granular gradient control.
    """
    
    def __init__(
        self,
        model: DualHeadGPT2,
        safe_loader: DataLoader,
        harmful_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        lambda_adv: float = 1.0,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            model: DualHeadGPT2 model
            safe_loader: DataLoader for safe tasks
            harmful_loader: DataLoader for harmful queries
            optimizer: Optimizer for model parameters
            device: Device to train on
            lambda_adv: Weight for adversarial loss
            grad_accum_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.safe_loader = safe_loader
        self.harmful_loader = harmful_loader
        self.optimizer = optimizer
        self.device = device
        self.lambda_adv = lambda_adv
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        
        self.model.to(device)
        self.model.train()
        
        # Training history
        self.history = {
            'main_loss': [],
            'adversarial_loss': [],
            'total_loss': [],
        }
    
    def train_epoch(self, epoch: int, log_interval: int = 10) -> Dict[str, float]:
        """
        Train for one epoch with granular gradient steps.
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        # Create iterators that cycle
        safe_iter = iter(self.safe_loader)
        harmful_iter = iter(self.harmful_loader)
        
        # Determine number of steps (use minimum of both loaders)
        num_steps = min(len(self.safe_loader), len(self.harmful_loader))
        
        total_main_loss = 0.0
        total_adv_loss = 0.0
        total_loss = 0.0
        
        progress_bar = tqdm(range(num_steps), desc=f"Epoch {epoch}")
        
        for step in progress_bar:
            # Get batches
            try:
                safe_batch = next(safe_iter)
            except StopIteration:
                safe_iter = iter(self.safe_loader)
                safe_batch = next(safe_iter)
            
            try:
                harmful_batch = next(harmful_iter)
            except StopIteration:
                harmful_iter = iter(self.harmful_loader)
                harmful_batch = next(harmful_iter)
            
            # Move to device
            safe_input_ids = safe_batch['input_ids'].to(self.device)
            safe_attention_mask = safe_batch['attention_mask'].to(self.device)
            safe_labels = safe_batch['labels'].to(self.device)
            
            harmful_input_ids = harmful_batch['input_ids'].to(self.device)
            harmful_attention_mask = harmful_batch['attention_mask'].to(self.device)
            harmful_labels = harmful_batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # ===== MAIN HEAD FORWARD PASS =====
            # Forward through main head with safe data
            main_outputs = self.model.forward_main(
                input_ids=safe_input_ids,
                attention_mask=safe_attention_mask,
                labels=safe_labels,
            )
            main_loss = main_outputs['loss']
            
            # Backward for main loss (normal gradients)
            main_loss.backward(retain_graph=False)
            
            # ===== ADVERSARIAL HEAD FORWARD PASS =====
            # Forward through adversarial head with harmful data
            # Gradient reversal is applied inside forward_adversarial
            adv_outputs = self.model.forward_adversarial(
                input_ids=harmful_input_ids,
                attention_mask=harmful_attention_mask,
                labels=harmful_labels,
            )
            adv_loss = adv_outputs['loss']
            
            # Backward for adversarial loss (with gradient reversal)
            # The gradient reversal is already applied in forward_adversarial
            adv_loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Compute total loss (weighted)
            total_step_loss = main_loss.item() + self.lambda_adv * adv_loss.item()
            
            # Update running totals
            total_main_loss += main_loss.item()
            total_adv_loss += adv_loss.item()
            total_loss += total_step_loss
            
            # Update progress bar
            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    'main_loss': f'{main_loss.item():.4f}',
                    'adv_loss': f'{adv_loss.item():.4f}',
                    'total': f'{total_step_loss:.4f}',
                })
        
        # Final gradient step if needed
        if num_steps % self.grad_accum_steps != 0:
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Compute averages
        avg_main_loss = total_main_loss / num_steps
        avg_adv_loss = total_adv_loss / num_steps
        avg_total_loss = total_loss / num_steps
        
        # Update history
        self.history['main_loss'].append(avg_main_loss)
        self.history['adversarial_loss'].append(avg_adv_loss)
        self.history['total_loss'].append(avg_total_loss)
        
        return {
            'main_loss': avg_main_loss,
            'adversarial_loss': avg_adv_loss,
            'total_loss': avg_total_loss,
        }
    
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[str] = None,
        save_interval: int = 1,
        log_interval: int = 10,
    ):
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
            log_interval: Log metrics every N steps
        """
        for epoch in range(1, num_epochs + 1):
            losses = self.train_epoch(epoch, log_interval)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Main Loss: {losses['main_loss']:.4f}")
            print(f"  Adversarial Loss: {losses['adversarial_loss']:.4f}")
            print(f"  Total Loss: {losses['total_loss']:.4f}")
            
            # Save checkpoint
            if save_dir and epoch % save_interval == 0:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    losses['total_loss'],
                    checkpoint_path,
                )
                print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Save final model and training history
        if save_dir:
            final_model_path = os.path.join(save_dir, 'final_model.pt')
            torch.save(self.model.state_dict(), final_model_path)
            
            history_path = os.path.join(save_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            print(f"\nSaved final model to {final_model_path}")
            print(f"Saved training history to {history_path}")
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history.copy()

