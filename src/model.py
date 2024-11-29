import torch
import torch.nn as nn
import torch.nn.functional as F

class FloorplanModel(nn.Module):
    def __init__(self, text_input_dim=128, image_input_dim=(1, 128, 128), hidden_dim=256, output_dim=128):
        """
        Neural network for processing text embeddings and images.
        :param text_input_dim: Number of tokens in text embeddings.
        :param image_input_dim: Dimensions of the input images (C, H, W).
        :param hidden_dim: Size of the hidden layers in the network.
        :param output_dim: Size of the final output (task-specific).
        """
        super(FloorplanModel, self).__init__()
        
        # Text Processing Branch
        self.text_fc = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Image Processing Branch
        self.image_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # First convolution
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Second convolution
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Flatten(),  # Flatten the feature maps
            nn.Linear(32 * 32 * 32, hidden_dim),  # Fully connected layer
            nn.ReLU()
        )
        
        # Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_embeddings, images):
        """
        Forward pass for the model.
        :param text_embeddings: Tensor of shape (batch_size, text_input_dim).
        :param images: Tensor of shape (batch_size, 1, H, W).
        :return: Output tensor of shape (batch_size, output_dim).
        """
        # Text processing
        text_features = self.text_fc(text_embeddings)

        # Image processing
        image_features = self.image_cnn(images)

        # Combine features
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Pass through fusion layer
        output = self.fusion_layer(combined_features)
        return output


if __name__ == "__main__":
    # Example usage
    batch_size = 16
    text_input_dim = 128  # Number of tokens in the embeddings
    image_input_dim = (1, 128, 128)  # Grayscale image dimensions
    
    # Dummy data for testing
    text_embeddings = torch.rand(batch_size, text_input_dim)  # Random text embeddings
    images = torch.rand(batch_size, *image_input_dim)  # Random image tensors
    
    # Initialize the model
    model = FloorplanModel()
    
    # Forward pass
    output = model(text_embeddings, images)
    print(f"Output shape: {output.shape}")  # Expected: [batch_size, output_dim]
