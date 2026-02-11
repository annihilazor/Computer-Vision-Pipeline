"""
CNN Feature Extractor for R-CNN
Uses pre-trained models (AlexNet, VGG, ResNet) as backbone
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple
import cv2


class CNNFeatureExtractor:
    """Extract features from region proposals using pre-trained CNN"""

    def __init__(self, backbone: str = 'vgg16', device: str = 'cuda',
                 freeze_backbone: bool = True):
        """
        Args:
            backbone: 'alexnet', 'vgg16', 'resnet50', etc.
            device: 'cuda' or 'cpu'
            freeze_backbone: Whether to freeze backbone weights
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone

        # Load pre-trained model
        self.model = self._load_backbone(backbone)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Determine feature dimension
        self.feature_dim = self._get_feature_dim()

    def _load_backbone(self, backbone: str) -> nn.Module:
        """Load pre-trained backbone model"""
        if backbone == 'alexnet':
            model = models.alexnet(pretrained=True)
            # Remove classifier, keep only features
            feature_extractor = model.features
            # Add adaptive pooling to get fixed size output
            feature_extractor = nn.Sequential(
                feature_extractor,
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten(),
                model.classifier[:6]  # Keep first 6 layers of classifier (before final fc)
            )

        elif backbone == 'vgg16':
            model = models.vgg16(pretrained=True)
            # Remove final classification layer
            feature_extractor = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                model.classifier[:6]  # Keep dropout layers but remove final fc
            )

        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove final fc layer
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_extractor.add_module('flatten', nn.Flatten())

        elif backbone == 'resnet18':
            model = models.resnet18(pretrained=True)
            feature_extractor = nn.Sequential(*list(model.children())[:-1])
            feature_extractor.add_module('flatten', nn.Flatten())

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Freeze weights if specified
        if self.freeze_backbone:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        return feature_extractor

    def _get_feature_dim(self) -> int:
        """Determine feature dimension by forward pass"""
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            features = self.model(dummy_input)
        return features.shape[1]

    def extract_features_from_regions(self, image: np.ndarray,
                                      regions: List[List[int]],
                                      batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple region proposals

        Args:
            image: Input image (RGB format, numpy array)
            regions: List of [xmin, ymin, xmax, ymax]
            batch_size: Batch size for processing

        Returns:
            Feature matrix of shape (num_regions, feature_dim)
        """
        features_list = []

        # Process in batches
        for i in range(0, len(regions), batch_size):
            batch_regions = regions[i:i + batch_size]
            batch_images = []

            # Crop and preprocess each region
            for box in batch_regions:
                xmin, ymin, xmax, ymax = box
                # Ensure valid box
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax = min(image.shape[1], xmax)
                ymax = min(image.shape[0], ymax)

                # Crop region
                region_img = image[ymin:ymax, xmin:xmax]

                # Handle empty regions
                if region_img.size == 0:
                    region_img = np.zeros((224, 224, 3), dtype=np.uint8)

                # Preprocess
                region_tensor = self.transform(region_img)
                batch_images.append(region_tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_images).to(self.device)

            # Extract features
            with torch.no_grad():
                batch_features = self.model(batch_tensor)

            features_list.append(batch_features.cpu().numpy())

        # Concatenate all features
        all_features = np.vstack(features_list)
        return all_features

    def extract_single_feature(self, image: np.ndarray, box: List[int]) -> np.ndarray:
        """
        Extract feature from a single region

        Args:
            image: Input image (RGB)
            box: [xmin, ymin, xmax, ymax]

        Returns:
            Feature vector of shape (feature_dim,)
        """
        xmin, ymin, xmax, ymax = box
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax = min(image.shape[1], xmax)
        ymax = min(image.shape[0], ymax)

        # Crop and preprocess
        region_img = image[ymin:ymax, xmin:xmax]

        if region_img.size == 0:
            region_img = np.zeros((224, 224, 3), dtype=np.uint8)

        region_tensor = self.transform(region_img).unsqueeze(0).to(self.device)

        # Extract feature
        with torch.no_grad():
            feature = self.model(region_tensor)

        return feature.cpu().numpy().squeeze()

    def save_model(self, path: str):
        """Save feature extractor"""
        torch.save({
            'backbone': self.backbone_name,
            'state_dict': self.model.state_dict(),
            'feature_dim': self.feature_dim
        }, path)

    def load_model(self, path: str):
        """Load feature extractor"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.feature_dim = checkpoint['feature_dim']
        self.model.eval()


class FineTunedCNN(nn.Module):
    """Fine-tuned CNN for R-CNN (optional alternative to frozen backbone)"""

    def __init__(self, backbone: str = 'vgg16', num_classes: int = 20,
                 feature_dim: int = 4096):
        super().__init__()

        self.backbone_name = backbone

        # Load pre-trained backbone
        if backbone == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            self.features = vgg.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.feature_extractor = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
            )

        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, feature_dim),
                nn.ReLU(True),
                nn.Dropout(),
            )

        # Classification head (for fine-tuning)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        """
        Forward pass

        Args:
            x: Input tensor (B, C, H, W)
            return_features: If True, return features instead of logits

        Returns:
            Logits or features
        """
        x = self.features(x)

        if hasattr(self, 'avgpool'):
            x = self.avgpool(x)

        x = torch.flatten(x, 1)
        features = self.feature_extractor(x)

        if return_features:
            return features

        logits = self.classifier(features)
        return logits
