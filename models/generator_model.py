import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class UpSampleBlock(nn.Module):
    """
    Upsampling block for the U-Net generator using interpolation followed by convolution.

    Consists of Upsample -> Conv2d -> BatchNorm -> Activation -> Conv2d -> BatchNorm -> Activation

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        use_batchnorm (bool): Whether to use batch normalization.
        interpolation_mode (str): Interpolation mode for upsampling.
                                 Options: 'nearest', 'bilinear', 'bicubic'.
        activation (callable): Activation function for upsampling.
    """
    def __init__(self, in_channel, skip_in_channel, out_channel, number_of_convolutions=2, use_batchnorm=True, interpolation_mode='bilinear', activation=nn.LeakyReLU()):
        super(UpSampleBlock, self).__init__()
        
        self.interpolation_mode = interpolation_mode
        if interpolation_mode == 'conv':
            self.transposed_conv = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1)

        # Convolution after upsampling
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel) if use_batchnorm else nn.Identity()
        self.activation1 = activation

        self.skip_conv = nn.Conv2d(in_channel + skip_in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.skip_bn = nn.BatchNorm2d(out_channel) if use_batchnorm else nn.Identity()
        self.skip_activation = activation

        self.number_of_convolutions = number_of_convolutions

        # Second convolution (after concatenation with skip connection)
        for i in range(number_of_convolutions):
            setattr(self, f"conv{i+3}", nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            setattr(self, f"bn{i+3}", nn.BatchNorm2d(out_channel) if use_batchnorm else nn.Identity())
            setattr(self, f"activation{i+3}", activation)

    def forward(self, x, skip_features):
        if self.interpolation_mode == 'conv':
            x = self.transposed_conv(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.interpolation_mode, align_corners=False if self.interpolation_mode != 'nearest' else None)
        # apply convolution after upsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip_features], dim=1)

        x = self.skip_conv(x)
        x = self.skip_bn(x)
        x = self.skip_activation(x)

        residual_x = x
        # Apply convolutions with residual connections
        for i in range(3, self.number_of_convolutions + 3):
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"bn{i}")(x)
            x = getattr(self, f"activation{i}")(x)

            x = x + residual_x # residual connection
            residual_x = x

        return x


class UNetGenerator(nn.Module):
    """
    U-Net Generator for image-to-image translation tasks.

    Features:
    - Flexible input/output channels for various tasks (e.g., 1->3 for colorization)
    - Skip connections to preserve spatial information
    - Interpolation-based upsampling to avoid checkerboard artifacts
    - Configurable depth to handle different image resolutions

    Args:
        in_channels (int): Number of input image channels.
        out_channels (int): Number of output image channels.
        init_features (int): Number of features in the first layer, doubles with each downsampling.
        depth (int): Depth of the U-Net, number of downsampling operations.
        use_batchnorm (bool): Whether to use batch normalization.
        use_maxpool (bool): Whether to use max pooling for downsampling.
        interpolation_mode (str): Mode for upsampling interpolation ('nearest', 'bilinear', 'bicubic').
        activation (callable): Activation function for downsampling and upsampling.
        final_activation (callable): Activation function for the final layer.
    """
    def __init__(self, in_channels=1, out_channels=3, use_batchnorm=True, use_maxpool=True, interpolation_mode='bilinear', activation=nn.LeakyReLU(), final_activation=nn.Tanh()):
        super(UNetGenerator, self).__init__()

        self.interpolation_mode = interpolation_mode
        # Encoder (downsampling) path
        self.encoder = resnet34(weights=ResNet34_Weights.DEFAULT)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512) if use_batchnorm else nn.Identity(),
            activation,
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512) if use_batchnorm else nn.Identity(),
            activation,
        )
        self.d_layer1 = UpSampleBlock(512, 256, 256, number_of_convolutions=16, use_batchnorm=use_batchnorm, interpolation_mode=interpolation_mode, activation=activation)
        self.d_layer2 = UpSampleBlock(256, 128, 128, number_of_convolutions=12, use_batchnorm=use_batchnorm, interpolation_mode=interpolation_mode, activation=activation)
        self.d_layer3 = UpSampleBlock(128, 64, 64, number_of_convolutions=8, use_batchnorm=use_batchnorm, interpolation_mode=interpolation_mode, activation=activation)
        self.d_layer4 = UpSampleBlock(64, 64, 64, number_of_convolutions=8, use_batchnorm=use_batchnorm, interpolation_mode=interpolation_mode, activation=activation)
        self.d_layer5 = UpSampleBlock(64, 0, 64, number_of_convolutions=6, use_batchnorm=use_batchnorm, interpolation_mode=interpolation_mode, activation=activation)

        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.final_activation = final_activation

    def forward(self, x):
        # Store skip connections
        skip_connections = []
        x = x.repeat(1, 3, 1, 1)
        # Encoder path
        x = self.encoder.conv1(x)    # 64, 32, 32
        x = self.encoder.bn1(x)      # 64, 32, 32
        x = self.encoder.relu(x)     # 64, 32, 32
        skip_connections.append(x)
        x = self.encoder.maxpool(x)  # 64, 16, 16
        x = self.encoder.layer1(x)   # 64, 16, 16
        skip_connections.append(x)
        x = self.encoder.layer2(x)   # 128, 8, 8
        skip_connections.append(x)
        x = self.encoder.layer3(x)   # 256, 4, 4
        skip_connections.append(x)
        x = self.encoder.layer4(x)   # 512, 2, 2

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        x = self.d_layer1(x, skip_connections[3])
        x = self.d_layer2(x, skip_connections[2])
        x = self.d_layer3(x, skip_connections[1])
        x = self.d_layer4(x, skip_connections[0])
        x = self.d_layer5(x, torch.tensor([], device=x.device))

        # Final convolution and activation
        x = self.final_conv(x)
        x = self.final_activation(x)

        return x


def print_unet_dimensions(input_size=(64,64), in_channels=1, out_channels=3):
    """
    Helper function to print the dimensions of each layer in the U-Net.

    Args:
        input_size (tuple): Input image dimensions (height, width).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    print(f"U-Net Dimensions for {input_size[0]}x{input_size[1]} image:")

    # Calculate appropriate depth
    min_dim = min(input_size)
    max_depth = 0
    while min_dim >= 8:  # Ensure smallest feature map is at least 8x8
        min_dim = min_dim // 2
        max_depth += 1

    print(f"Recommended max depth: {max_depth}")

    # Print layer dimensions
    current_h, current_w = input_size
    features = 64  # starting features

    print(f"Input: {in_channels}x{current_h}x{current_w}")
    print(f"Initial: {features}x{current_h}x{current_w}")

    # Encoder
    for i in range(max_depth):
        current_h, current_w = current_h // 2, current_w // 2
        features *= 2
        print(f"Encoder {i + 1}: {features}x{current_h}x{current_w}")

    # Decoder
    for i in range(max_depth):
        features //= 2
        current_h, current_w = current_h * 2, current_w * 2
        print(f"Decoder {i + 1}: {features}x{current_h}x{current_w}")

    print(f"Output: {out_channels}x{current_h}x{current_w}")


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # For colorization task
    model = UNetGenerator(in_channels=1, out_channels=3, final_activation=nn.Tanh(), activation=nn.LeakyReLU())

    # Print model structure
    print(model)
    print(f"Number of trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Encoder number of parameters
    print(f"Encoder number of parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")

    # Bottleneck number of parameters
    print(f"Bottleneck number of parameters: {sum(p.numel() for p in model.bottleneck.parameters()):,}")

    # Decoder number of parameters
    decoder_number_of_parameters = 0
    decoder_number_of_parameters += sum(p.numel() for p in model.d_layer1.parameters())
    decoder_number_of_parameters += sum(p.numel() for p in model.d_layer2.parameters())
    decoder_number_of_parameters += sum(p.numel() for p in model.d_layer3.parameters())
    decoder_number_of_parameters += sum(p.numel() for p in model.d_layer4.parameters())
    decoder_number_of_parameters += sum(p.numel() for p in model.d_layer5.parameters())
    decoder_number_of_parameters += sum(p.numel() for p in model.final_conv.parameters())
    print(f"Decoder number of parameters: {decoder_number_of_parameters:,}")
    
    # Check dimensions
    print_unet_dimensions()

    # Test with sample input
    x = torch.randn(1, 1, 64, 64)  # Batch size 1, 1 channel (grayscale), 64x64 image
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Test backward pass
    print("\nTesting backward pass...")

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to(device)
        x = x.to(device)

    # Create a fake target (simulating a real colorized image)
    target = torch.randn(1, 3, 64, 64)
    if torch.cuda.is_available():
        target = target.to(device)

    # Define a loss function
    criterion = nn.MSELoss()

    # Forward pass
    output = model(x)

    # Calculate loss
    loss = criterion(output, target)
    print(f"Loss value: {loss.item()}")

    # Backward pass
    loss.backward()

    # Check that gradients have been calculated
    has_gradients = all(param.grad is not None for param in model.parameters() if param.requires_grad)
    print(f"All parameters have gradients: {has_gradients}")

    # Print gradient statistics for verification
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            # Print first few parameter gradients as examples
            if param_count < 3:
                print(f"Gradient norm for {name}: {param_norm}")
            param_count += 1

    total_norm = total_norm ** 0.5
    print(f"Total gradient norm: {total_norm}")

    # Reset gradients
    model.zero_grad()
    print("Backward pass completed successfully")
