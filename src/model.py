import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import T5Model

class AthenaModel(nn.Module):
    def __init__(self, text_embedding_dim=512, image_feature_dim=512, latent_dim=1024):
        super(AthenaModel, self).__init__()

        # 1. Text Encoder (T5 Model)
        self.text_encoder = T5Model.from_pretrained("t5-small")
        self.text_projection = nn.Linear(512, text_embedding_dim)

        # 2. Image Encoder (Modified ResNet18 for 1-channel input)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 1-channel (grayscale) input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
        self.image_projection = nn.Linear(512, image_feature_dim)

        # 3. Feature Fusion
        self.fusion_layer = nn.Linear(text_embedding_dim + image_feature_dim, latent_dim)
        self.fusion_activation = nn.ReLU()

                # 4. Decoder (Image Generation)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # Final output
            nn.Tanh()  # Output normalized grayscale image
        )


    def forward(self, text_tokens, image_tensor):
        text_features = self.text_encoder.encoder(input_ids=text_tokens).last_hidden_state
        text_features = text_features.mean(dim=1)
        text_features = self.text_projection(text_features)

        # Process the grayscale image through modified ResNet18
        image_features = self.image_encoder(image_tensor)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_projection(image_features)

        # Merge text and image features
        fused_features = torch.cat((text_features, image_features), dim=1)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.fusion_activation(fused_features)

        # Reshape for decoder and generate image
        fused_features = fused_features.view(fused_features.size(0), -1, 1, 1)
        generated_image = self.decoder(fused_features)

        return generated_image

# Test the model
if __name__ == "__main__":
    model = AthenaModel()
    text_input = torch.randint(0, 100, (2, 128))  # Simulated tokenized text
    image_input = torch.randn(2, 1, 256, 256)  # Simulated grayscale images

    output = model(text_input, image_input)
    print(f"Generated image shape: {output.shape}")  # Should output: (2, 1, 256, 256)
