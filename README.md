# Mitigating LLM Dual-Use with Contextual Latent Adversarial Unlearning

This project implements a dual-head adversarial training setup for GPT-2 Medium to mitigate dual-use capabilities by excising harmful behavior at the representation level, rather than at the output level.

## Overview

The approach uses a dual-head architecture where:
- **Shared trunk**: First `n` layers of GPT-2
- **Main head**: Standard LM head trained with safe task data
- **Adversarial head**: LM head trained to generate harmful queries, with gradient reversal to suppress dual-use information in the shared representations

By targeting the latent geometry itself, we aim to achieve more robust safety compared to prompt-based guardrails.

## Project Structure

```
CS230Proj/
├── src/
│   ├── model.py              # GPT-2 with dual-head architecture
│   ├── training.py           # Adversarial training loop (granular gradient steps)
│   ├── data.py               # Data loading and preprocessing
│   ├── evaluation.py         # Evaluation metrics and jailbreak testing
│   └── utils.py              # Helper functions
├── configs/
│   └── config.yaml           # Training hyperparameters
├── data/
│   ├── safe/                 # CNN/DailyMail, Natural Questions
│   └── harmful/              # AdvBench/JailbreakBench prompts
├── experiments/
│   └── layer_experiments/    # Results for each layer experiment
├── scripts/
│   ├── train.py              # Main training script
│   └── evaluate.py           # Evaluation script
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
cd CS230Proj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have GPU access (CUDA) for training.

## Usage

### Training

Train models for different branch layers:

```bash
# Train all layers specified in config
python scripts/train.py --config configs/config.yaml

# Train a specific layer
python scripts/train.py --config configs/config.yaml --layer 12
```

The training script will:
- Load safe and harmful data
- Create dual-head models for each specified layer
- Train with adversarial unlearning
- Save checkpoints to `experiments/layer_experiments/layer_{n}/`

### Evaluation

Evaluate trained models:

```bash
# Evaluate all layers specified in config
python scripts/evaluate.py --config configs/config.yaml

# Evaluate a specific layer
python scripts/evaluate.py --config configs/config.yaml --layer 12

# Evaluate with a specific checkpoint
python scripts/evaluate.py --config configs/config.yaml --layer 12 --checkpoint experiments/layer_experiments/layer_12/checkpoint_epoch_3.pt
```

The evaluation script will:
- Compute perplexity on safe tasks
- Measure dual-use suppression
- Test jailbreak robustness
- Generate plots and reports in `experiments/results/`

## Configuration

Edit `configs/config.yaml` to adjust:
- Model settings (branch layers, lambda_adv)
- Training hyperparameters (batch size, learning rate, epochs)
- Data paths and sample sizes
- Evaluation settings

## Key Features

1. **Granular Gradient Control**: Direct gradient manipulation without high-level trainers
2. **Autoregressive Adversarial Loss**: Adversarial head uses LM loss on harmful queries
3. **Layer-wise Experiments**: Test different branch points in the transformer
4. **Comprehensive Evaluation**: Metrics for both safety and utility

## Proof-of-Concept Scope

The initial implementation focuses on:
- 5 representative layers: 1, 6, 12, 18, 24
- Synthetic harmful queries (can be replaced with AdvBench/JailbreakBench)
- Basic jailbreak evaluation
- Trade-off analysis between utility and suppression

## Methodology

The training process:
1. Forward safe data through main head → compute main LM loss
2. Forward harmful queries through adversarial head → compute adversarial LM loss
3. Apply gradient reversal to adversarial gradients
4. Update shared trunk weights to preserve utility while suppressing dual-use

## Evaluation Metrics

- **Safe Task Utility**: Perplexity on safe datasets
- **Dual-Use Suppression**: Adversarial head loss on harmful queries
- **Jailbreak Robustness**: Success rate of jailbreak attempts
- **Trade-off Curves**: Utility vs. suppression across layers

## References

- Ravfogel et al. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. ACL.
- Ganin & Lempitsky (2015). Unsupervised Domain Adaptation by Backpropagation. ICML.
- Eldan et al. (2023). Erasing Concepts from LLMs with Low-Rank Adaptation. arXiv.
- Anthropic (2024). Towards Monosemanticity: Decomposing Language Models with Sparse Autoencoders. arXiv.

## License

This project is for research purposes.

