# Image Colorization with GANs

A deep learning project that implements an image colorization system using Generative Adversarial Networks (GANs). This project can transform grayscale images into realistic colored images using a U-Net generator and PatchGAN discriminator architecture.

## 🎯 Project Overview

This project implements an image-to-image translation system that learns to colorize grayscale images. It uses a conditional GAN architecture with:
- **Generator**: U-Net architecture with skip connections and FiLM conditioning
- **Discriminator**: PatchGAN architecture for high-quality local detail preservation
- **Training**: WGAN-GP with additional L1 and mode-seeking losses

## 🏗️ Architecture

### Generator (UNetGenerator)
- **U-Net Structure**: Encoder-decoder architecture with skip connections
- **FiLM Conditioning**: Feature-wise linear modulation for conditional generation
- **Upsampling**: Multiple interpolation modes (bilinear, nearest, bicubic, transposed convolution)
- **Residual Connections**: Within upsampling blocks for better gradient flow
- **ResNet34 Encoder**: Pre-trained backbone for feature extraction

### Discriminator (PatchGAN)
- **PatchGAN70**: 70x70 receptive field for local detail discrimination
- **Spectral Normalization**: Optional for training stability
- **Group Normalization**: Alternative to batch normalization

## 📁 Project Structure

```
image_colorizer/
├── loaders/                   # Data loading utilities
│   ├── data_loader.py        # Tiny ImageNet dataset loader
│   └── __init__.py
├── models/                    # Neural network models
│   ├── generator_model.py    # U-Net generator implementation
│   ├── discriminator.py      # PatchGAN discriminator
│   └── __init__.py
├── train/                     # Training infrastructure
│   ├── runner.py             # Main training script
│   ├── trainer.py            # Training loop and logic
│   └── __init__.py
├── utils/                     # Utility functions
│   ├── config.py             # Configuration classes
│   ├── kaggle_utils.py       # Dataset download utilities
│   ├── logging.py            # TensorBoard logging
│   └── metrics.py            # Evaluation metrics (PSNR, SSIM)
├── playground.ipynb          # Interactive notebook for experimentation
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd image_colorizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   python utils/kaggle_utils.py
   ```
   
   **Note**: You'll need to configure Kaggle API credentials first:
   - Install Kaggle CLI: `pip install kaggle`
   - Download your API token from [Kaggle Settings](https://www.kaggle.com/settings/account)
   - Place `kaggle.json` in `~/.kaggle/`

### Training

1. **Configure training parameters** in `train/runner.py`:
   ```python
   config = TrainingArgs(
       epochs=12,
       batch_size=16,
       image_size=128,
       device="cuda",  # or "cpu"
       data_path="data_kaggle/data",
       # ... other parameters
   )
   ```

2. **Start training**:
   ```bash
   python train/runner.py
   ```

3. **Monitor training** with TensorBoard:
   ```bash
   tensorboard --logdir logs_big_images_final_v5_tmp
   ```

## ⚙️ Configuration

### Training Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `epochs` | Number of training epochs | 12 |
| `batch_size` | Training batch size | 16 |
| `image_size` | Input image resolution | 128 |
| `device` | Training device | "cpu" |
| `generator_lr` | Generator learning rate | 0.0001 |
| `discriminator_lr` | Discriminator learning rate | 0.0001 |
| `lambda_gp` | Gradient penalty weight | 10 |
| `lambda_l1` | L1 loss weight | 100 |
| `lambda_mode_seeking` | Mode-seeking loss weight | 1 |
| `critic_n` | Critic updates per generator update | 5 |

### Model Architecture Options

- **Generator**: U-Net with ResNet34 encoder, FiLM conditioning
- **Discriminator**: PatchGAN variants (70x70 or 34x34 receptive field)
- **Upsampling**: Multiple interpolation modes supported
- **Normalization**: BatchNorm, GroupNorm, or Spectral Normalization

## 🎨 Usage Examples

### Interactive Notebook
Use `playground.ipynb` for experimentation and visualization:

```python
# Load pre-trained models
generator = UNetGenerator(...)
generator.load_state_dict(torch.load('checkpoints_good/generator_epoch_11.pth'))

# Colorize a grayscale image
with torch.no_grad():
    colored = generator(grayscale_image)
```

## 📊 Training Details

### Loss Functions
- **Adversarial Loss**: WGAN-GP for stable training
- **L1 Loss**: Pixel-wise reconstruction loss
- **Mode-Seeking Loss**: Prevents mode collapse
- **Gradient Penalty**: Enforces Lipschitz constraint

### Training Strategy
- **Critic Updates**: 5 discriminator updates per generator update
- **Learning Rate Scheduling**: Cosine annealing for both networks
- **Gradient Accumulation**: 4 steps for effective larger batch sizes
- **Validation**: Regular evaluation with PSNR metric

### Monitoring
- **TensorBoard Logging**: Loss curves, generated samples, metrics
- **Checkpointing**: Model saves every epoch
- **Validation Metrics**: PSNR

## 🔧 Customization

### Adding New Datasets
Extend `TinyImageNetDataset` in `loaders/data_loader.py`:

```python
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Implement your dataset logic
        pass
    
    def __getitem__(self, idx):
        # Return {'image': grayscale, 'target': colored}
        pass
```

### Modifying Architecture
- **Generator**: Adjust U-Net depth, add/remove FiLM layers
- **Discriminator**: Change receptive field size, normalization type
- **Loss Functions**: Modify loss weights or add custom losses

## 📈 Performance

### Pre-trained Models
- Generator and discriminator checkpoints for epochs 0-11
- Trained on 128x128 images with Tiny ImageNet dataset

### Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio for image quality

### Debuging
Everything is logged via tensorboard!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 References

- **U-Net**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **PatchGAN**: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- **WGAN-GP**: [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- **FiLM**: [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- **MSGAN**: [Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis](https://arxiv.org/abs/1903.05628)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is for research and educational purposes. For production use, consider additional optimizations and robust evaluation protocols.
