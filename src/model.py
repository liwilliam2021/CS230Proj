"""
Dual-head GPT-2 model with adversarial unlearning architecture.
"""
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for adversarial training (Ganin & Lempitsky, 2015)."""
    
    def __init__(self, lambda_adv=1.0):
        super().__init__()
        self.lambda_adv = lambda_adv
    
    def forward(self, x):
        return x
    
    def backward(self, grad_output):
        return -self.lambda_adv * grad_output


class GradientReversalFunction(torch.autograd.Function):
    """Autograd function for gradient reversal."""
    
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def gradient_reversal(x, lambda_adv=1.0):
    """Apply gradient reversal to tensor."""
    return GradientReversalFunction.apply(x, lambda_adv)


class DualHeadGPT2(nn.Module):
    """
    GPT-2 with dual-head architecture for adversarial unlearning.
    
    Architecture:
    - Shared trunk: First n layers of GPT-2
    - Main head: Standard LM head (next token prediction)
    - Adversarial head: LM head that attempts to generate harmful queries
    """
    
    def __init__(self, model_name='gpt2-medium', branch_layer=12, lambda_adv=1.0):
        """
        Args:
            model_name: HuggingFace model name (default: 'gpt2-medium')
            branch_layer: Layer at which to branch (1-24 for GPT-2 Medium)
            lambda_adv: Weight for adversarial loss (gradient reversal coefficient)
        """
        super().__init__()
        
        # Load base GPT-2 model
        self.base_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.base_model.config
        self.branch_layer = branch_layer
        self.lambda_adv = lambda_adv
        
        # Get the transformer blocks
        self.transformer = self.base_model.transformer
        self.num_layers = len(self.transformer.h)
        
        assert 1 <= branch_layer <= self.num_layers, \
            f"branch_layer must be between 1 and {self.num_layers}"
        
        # Main head (standard LM head)
        self.main_head = self.base_model.lm_head
        
        # Adversarial head (separate LM head for harmful query generation)
        # Initialize from main head but will be trained adversarially
        self.adversarial_head = nn.Linear(
            self.config.n_embd,
            self.config.vocab_size,
            bias=False
        )
        # Initialize adversarial head with main head weights
        self.adversarial_head.weight.data = self.main_head.weight.data.clone()
        
        # Embedding layer (shared)
        self.wte = self.transformer.wte
        self.wpe = self.transformer.wpe
        
    def get_hidden_states(self, input_ids, attention_mask=None, past_key_values=None):
        """
        Forward pass through shared trunk (first branch_layer layers).
        
        Returns:
            hidden_states: Hidden states at branch layer
            past_key_values: Past key values for generation
        """
        # Get embeddings
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
        
        device = input_ids.device
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        # Apply dropout
        hidden_states = self.transformer.drop(hidden_states)
        
        # Forward through first branch_layer transformer blocks
        past_key_values = past_key_values or [None] * self.num_layers
        use_cache = past_key_values[0] is not None
        
        for i in range(self.branch_layer):
            layer = self.transformer.h[i]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i] if use_cache else None,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
        
        return hidden_states, past_key_values[:self.branch_layer] if use_cache else None
    
    def forward_main(self, input_ids, attention_mask=None, labels=None, past_key_values=None):
        """
        Forward pass through main head (standard LM objective).
        
        Returns:
            logits: Main head logits
            loss: Main LM loss (if labels provided)
        """
        hidden_states, past_key_values = self.get_hidden_states(
            input_ids, attention_mask, past_key_values
        )
        
        # Continue through remaining layers
        for i in range(self.branch_layer, self.num_layers):
            layer = self.transformer.h[i]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i] if past_key_values else None,
                use_cache=past_key_values is not None,
            )
            hidden_states = layer_outputs[0]
        
        # Apply layer norm
        hidden_states = self.transformer.ln_f(hidden_states)
        
        # Main head
        logits = self.main_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
        }
    
    def forward_adversarial(self, input_ids, attention_mask=None, labels=None, past_key_values=None):
        """
        Forward pass through adversarial head (harmful query generation).
        Applies gradient reversal to hidden states.
        
        Returns:
            logits: Adversarial head logits
            loss: Adversarial LM loss (if labels provided)
        """
        hidden_states, past_key_values = self.get_hidden_states(
            input_ids, attention_mask, past_key_values
        )
        
        # Apply gradient reversal
        hidden_states = gradient_reversal(hidden_states, self.lambda_adv)
        
        # Continue through remaining layers
        for i in range(self.branch_layer, self.num_layers):
            layer = self.transformer.h[i]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i] if past_key_values else None,
                use_cache=past_key_values is not None,
            )
            hidden_states = layer_outputs[0]
        
        # Apply layer norm
        hidden_states = self.transformer.ln_f(hidden_states)
        
        # Adversarial head
        logits = self.adversarial_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
        }
    
    def forward(self, input_ids, attention_mask=None, labels=None, use_adversarial=False):
        """
        Forward pass - routes to main or adversarial head.
        
        Args:
            use_adversarial: If True, use adversarial head; else use main head
        """
        if use_adversarial:
            return self.forward_adversarial(input_ids, attention_mask, labels)
        else:
            return self.forward_main(input_ids, attention_mask, labels)
    
    def generate(self, input_ids, attention_mask=None, max_length=100, use_adversarial=False, **kwargs):
        """
        Generate text using either main or adversarial head.
        """
        model = self
        if use_adversarial:
            # Temporarily replace main head with adversarial head for generation
            original_head = model.main_head
            model.main_head = model.adversarial_head
        
        # Use base model's generation method but with our forward
        # For simplicity, we'll use a basic generation loop
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.size(1)):
            # Get logits
            if use_adversarial:
                outputs = model.forward_adversarial(generated, attention_mask)
            else:
                outputs = model.forward_main(generated, attention_mask)
            
            logits = outputs['logits'][:, -1, :]
            
            # Sample next token
            if 'temperature' in kwargs:
                logits = logits / kwargs['temperature']
            if 'top_k' in kwargs:
                # Top-k sampling
                top_k = kwargs['top_k']
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
                ], dim=1)
        
        if use_adversarial:
            model.main_head = original_head
        
        return generated

