from train.trainer import MyTrainer
from utils.config import TrainingArgs
import torch

if __name__ == "__main__":
    # Load the config
    config = TrainingArgs(
        epochs=12,
        batch_size=16,
        image_size=128,
        device="cpu",
        data_path="data_kaggle/data",
        generator__init__args={'interpolation_mode': 'bilinear'},
        discriminator__init__args={'in_channels': 4, 
                                    'base': 64, 
                                    'spectral': False},
        generator_lr=0.0001,
        discriminator_lr=0.0001,
        generator_beta1=0.0,
        generator_beta2=0.9,
        discriminator_beta1=0.0,
        discriminator_beta2=0.9,
        lambda_gp=10,
        lambda_l1=100,
        lambda_l1_low=100,
        lambda_mode_seeking=1,
        lambda_mode_seeking_low=1,
        critic_n=5,
        val_freq=10,
        log_dir='logs_big_images_final_v5_tmp',
        accumulation_steps=4,
    )
    trainer = MyTrainer(config)
    # Train the model
    trainer.train()
    print("Training complete")