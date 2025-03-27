import torch
import torch.nn.functional as F
import torch.nn as nn


class DownSampleBlock(nn.Module):
    """
    Downsampling block for the U-Net generator.

    Consists of Conv2d -> BatchNorm -> Activation -> Conv2d -> BatchNorm -> Activation
    followed by a MaxPool2d for downsampling.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        use_batchnorm (bool): Whether to use batch normalization.
        use_maxpool (bool): Whether to use max pooling for downsampling.
                           If False, will use strided convolution instead.
        activation (callable): Activation function for downsampling.
    """
    def __init__(self, in_channel, out_channel, use_batchnorm=True, use_maxpool=True, activation=nn.LeakyReLU()):
        super(DownSampleBlock, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel) if use_batchnorm else nn.Identity()
        self.activation1 = activation


        # Second convolution block
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel) if use_batchnorm else nn.Identity()
        self.activation2 = activation

        # Downsampling
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
             # Strided convolution alternative for downsampling
            self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        features = self.activation2(x)

        x_down = self.downsample(features)

        return x_down, features


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
    def __init__(self, in_channel, out_channel, use_batchnorm=True, interpolation_mode='bilinear', activation=nn.LeakyReLU()):
        super(UpSampleBlock, self).__init__()

        self.interpolation_mode = interpolation_mode

        # Convolution after upsampling
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel) if use_batchnorm else nn.Identity()
        self.activation1 = activation

        # Second convolution (after concatenation with skip connection)
        self.conv2 = nn.Conv2d(in_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel) if use_batchnorm else nn.Identity()
        self.activation2 = activation

    def forward(self, x, skip_features):
        x = F.interpolate(x, scale_factor=2, mode=self.interpolation_mode, align_corners=False if self.interpolation_mode != 'nearest' else None)

        # apply convolution after upsampling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip_features], dim=1)

        # Apply second convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)

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
    def __init__(self, in_channels=1, out_channels=3, init_features=64, depth=4, use_batchnorm=True, use_maxpool=True, interpolation_mode='bilinear', activation=nn.LeakyReLU(), final_activation=nn.Tanh()):
        super(UNetGenerator, self).__init__()

        self.depth = depth

        # Encoder (downsampling) path
        self.down_blocks = nn.ModuleList()

        # Initial block doesn't downsample
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(init_features) if use_batchnorm else nn.Identity(),
            activation,
        )


        # Downsample blocks
        in_features = init_features
        for i in range(depth):
            out_features = in_features * 2
            self.down_blocks.append(
                DownSampleBlock(in_features, out_features, use_batchnorm, use_maxpool, activation)
            )
            in_features = out_features

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features) if use_batchnorm else nn.Identity(),
            activation,
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features) if use_batchnorm else nn.Identity(),
            activation,
        )

        # Decoder (upsampling) path
        self.up_blocks = nn.ModuleList()

        for i in range(depth):
            out_features = in_features // 2
            self.up_blocks.append(
                UpSampleBlock(in_features, out_features, use_batchnorm, interpolation_mode, activation)
            )
            in_features = out_features

        # Final layer
        self.final_conv = nn.Conv2d(in_features, out_channels, kernel_size=1)

        self.final_activation = final_activation

    def forward(self, x):
        # Initial features
        x = self.initial_conv(x)

        # Store skip connections
        skip_connections = []

        # Encoder path
        for down_block in self.down_blocks:
            x, features = down_block(x)
            skip_connections.append(features)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections (in reverse order)
        for i, up_block in enumerate(self.up_blocks):
            skip_features = skip_connections[-(i + 1)]
            x = up_block(x, skip_features)

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
    model = UNetGenerator(in_channels=1, out_channels=3, depth=4, final_activation=nn.Tanh(), activation=nn.LeakyReLU())

    # Print model structure
    print(model)
    print(f"Number of trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
