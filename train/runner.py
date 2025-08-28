from train.trainer import MyTrainer
from utils.config import TrainingArgs

if __name__ == "__main__":
    # Load the config
    config = TrainingArgs(
        epochs=5,
        batch_size=128,
        image_size=64,
        device="mps",
        data_path="data/tiny-imagenet-200",
        generator__init__args={'interpolation_mode': 'bilinear'},
        discriminator__init__args={'in_channels': 4, 
                                   'base': 64, 
                                   'spectral': True},
        generator_lr=0.0002,
        discriminator_lr=0.0002,
        generator_beta1=0.0,
        generator_beta2=0.9,
        discriminator_beta1=0.0,
        discriminator_beta2=0.9,
        lambda_gp=10,
        lambda_l1=20,
        critic_n=5,
        val_freq=10,
        log_dir='logs_test',
    )
    # Create the trainer
    trainer = MyTrainer(config)
    # Train the model
    trainer.train()
    print("Training complete")