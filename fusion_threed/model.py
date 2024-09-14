import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class SelfAttention2d(nn.Module):
    """
    Self-Attention layer for 2D feature maps.

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels: int):
        super(SelfAttention2d, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Self-Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        batch_size, C, H, W = x.size()
        logger.debug(f"Self-Attention2D input shape: {x.shape}")

        proj_query = (
            self.query_conv(x)
            .view(batch_size, -1, H * W)
            .permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        logger.debug(f"Self-Attention2D output shape: {out.shape}")
        return out


class SelfAttention3d(nn.Module):
    """
    Self-Attention layer for 3D feature maps.

    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels: int):
        super(SelfAttention3d, self).__init__()
        self.query_conv = nn.Conv3d(
            in_channels, in_channels // 8, kernel_size=1
        )
        self.key_conv = nn.Conv3d(
            in_channels, in_channels // 8, kernel_size=1
        )
        self.value_conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Self-Attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        batch_size, C, D, H, W = x.size()
        logger.debug(f"Self-Attention3D input shape: {x.shape}")

        proj_query = (
            self.query_conv(x)
            .view(batch_size, -1, D * H * W)
            .permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(batch_size, -1, D * H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(
            batch_size, -1, D * H * W
        )

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, D, H, W)
        out = self.gamma * out + x
        logger.debug(f"Self-Attention3D output shape: {out.shape}")
        return out


class Diffusion3DGenerator(nn.Module):
    """
    Diffusion model that generates a 3D scene from an input image.

    This model uses an encoder to process the input image, incorporates time embeddings
    for diffusion steps, and employs a decoder with self-attention mechanisms to generate
    a 3D voxel grid representing the scene.

    Args:
        image_size (int): Size of the input image (assumed square).
        voxel_size (int): Size of the output voxel grid (assumed cubic).
        num_timesteps (int): Number of diffusion timesteps.
        hidden_dim (int): Dimension of hidden layers.
    """

    def __init__(
        self,
        image_size: int,
        voxel_size: int,
        num_timesteps: int,
        hidden_dim: int = 64,
    ):
        super(Diffusion3DGenerator, self).__init__()
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        # Encoder to process input image
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, hidden_dim, kernel_size=4, stride=2, padding=1
            ),  # Downsample
            nn.ReLU(),
            SelfAttention2d(hidden_dim),
            nn.Conv2d(
                hidden_dim,
                hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            SelfAttention2d(hidden_dim * 2),
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            SelfAttention2d(hidden_dim * 4),
        )

        # Time embedding for diffusion steps
        self.time_embedding = nn.Embedding(
            num_timesteps, hidden_dim * 4
        )

        # Linear layer to map encoded features to decoder input
        self.fc = nn.Linear(
            (image_size // 8) * (image_size // 8) * hidden_dim * 4,
            hidden_dim * 4,
        )

        # Decoder to generate 3D voxel grid
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                hidden_dim * 4,
                hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 1 -> 2
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(),
            SelfAttention3d(hidden_dim * 4),
            nn.ConvTranspose3d(
                hidden_dim * 4,
                hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 2 -> 4
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(),
            SelfAttention3d(hidden_dim * 2),
            nn.ConvTranspose3d(
                hidden_dim * 2,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 4 -> 8
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            SelfAttention3d(hidden_dim),
            nn.ConvTranspose3d(
                hidden_dim,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 8 -> 16
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            SelfAttention3d(hidden_dim),
            nn.ConvTranspose3d(
                hidden_dim, 1, kernel_size=4, stride=2, padding=1
            ),  # 16 -> 32
            nn.Sigmoid(),  # Output voxel occupancy probabilities
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, image_size, image_size).
            t (torch.Tensor): Time step tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Output voxel grid of shape (batch_size, 1, voxel_size, voxel_size, voxel_size).
        """
        logger.info(
            f"Forward pass started with input shape {x.shape} and time steps {t}"
        )

        # Encode input image
        encoded = self.encoder(x)
        logger.debug(f"Encoded image shape: {encoded.shape}")

        # Flatten encoded features
        encoded = encoded.view(encoded.size(0), -1)
        logger.debug(
            f"Flattened encoded features shape: {encoded.shape}"
        )

        # Pass through linear layer
        encoded = self.fc(encoded)
        logger.debug(
            f"Features after linear layer shape: {encoded.shape}"
        )

        # Embed time steps and combine with encoded features
        t_emb = self.time_embedding(t)
        encoded = encoded + t_emb
        logger.debug(f"Combined features shape: {encoded.shape}")

        # Reshape for decoder input
        decoder_input = encoded.view(-1, self.hidden_dim * 4, 1, 1, 1)
        logger.debug(f"Decoder input shape: {decoder_input.shape}")

        # Decode to generate voxel grid
        output = self.decoder(decoder_input)
        logger.info(f"Output voxel grid shape: {output.shape}")

        return output


# # Example usage
# if __name__ == "__main__":
#     # Sample input image and time step
#     batch_size = 2
#     image_size = 64
#     voxel_size = 32
#     num_timesteps = 1000

#     model = Diffusion3DGenerator(
#         image_size=image_size,
#         voxel_size=voxel_size,
#         num_timesteps=num_timesteps,
#     )
#     sample_image = torch.randn(batch_size, 3, image_size, image_size)
#     sample_t = torch.randint(0, num_timesteps, (batch_size,))
#     output = model(sample_image, sample_t)
#     print(f"Generated voxel grid shape: {output.shape}")
