import dataclasses
import torch
from typing import Optional

@dataclasses.dataclass
class TrainingArgs:
    epochs: int                         # Number of epochs
    batch_size: int                     # Batch size
    image_size: int                     # Image size
    device: str                         # Device to use
    data_path: str                      # Path to the data
    generator__init__args: dict         # Generator __init__ arguments
    discriminator__init__args: dict     # Discriminator __init__ arguments
    generator_lr: float                 # Generator learning rate
    discriminator_lr: float             # Discriminator learning rate
    generator_beta1: float              # Generator beta1
    generator_beta2: float              # Generator beta2
    discriminator_beta1: float          # Discriminator beta1
    discriminator_beta2: float          # Discriminator beta2
    lambda_gp: float                    # Lambda GP
    lambda_l1: float                    # Lambda L1
    lambda_l1_low: float                # Lambda L1 Low
    lambda_mode_seeking: float          # Lambda Mode-Seeking
    lambda_mode_seeking_low: float      # Lambda Mode-Seeking Low
    critic_n: int                       # Number of critic updates per generator update
    val_freq: int                       # Number of validation in one epoch
    log_dir: str                        # Path to the log directory
    accumulation_steps: int             # Number of accumulation steps

@dataclasses.dataclass
class DiscriminatorLosses:
    d_loss: float
    em_loss: float
    gp_loss: float
    grad_norm: float
    distance: float

@dataclasses.dataclass
class GeneratorLosses:
    g_loss: float
    l1_loss: float
    mode_seeking_loss: float
    grad_norm: Optional[float]

@dataclasses.dataclass
class TrainLosses:
    generator_losses: GeneratorLosses
    discriminator_losses: DiscriminatorLosses

@dataclasses.dataclass
class ValidationLosses:
    generator_losses: GeneratorLosses
    distance: float
    psnr: float
    ssim: float
    sample_generated_images: torch.Tensor
    sample_targets: torch.Tensor
    sample_original_images: torch.Tensor