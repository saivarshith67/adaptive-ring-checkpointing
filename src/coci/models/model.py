import torch
import torch.nn as nn
import torchvision.models as models


def get_efficientnet_binary(pretrained=True):
    """
    Create EfficientNet-B0 model for binary classification.

    Args:
        pretrained (bool): Whether to load ImageNet pretrained weights.
            Defaults to True.

    Returns:
        EfficientNet model with binary classification head (2 classes).
    """
    if pretrained:
        model = models.efficientnet_b0(weights="DEFAULT")
    else:
        model = models.efficientnet_b0(weights=None)

    # Get the number of features from the classifier
    num_features = model.classifier[1].in_features

    # Replace classifier with Dropout + Linear for binary classification
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2), torch.nn.Linear(num_features, 2)
    )

    return model


def get_model(name, num_classes=100, pretrained=True):

    if name == "resnet18":
        weights = "DEFAULT" if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        weights = "DEFAULT" if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "mobilenet":
        weights = "DEFAULT" if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(1280, num_classes)

    elif name == "efficientnet_b0":
        model = get_efficientnet_binary(pretrained=pretrained)

        # Adjust for num_classes if needed
        if num_classes != 2:
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(in_features, num_classes)
            )

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


def get_multiframe_model(
    backbone: str = "efficientnet_b0",
    temporal_mode: str = "mean",
    num_classes: int = 2,
    hidden_size: int = 256,
    num_layers: int = 1,
    pretrained: bool = True,
):
    """
    Create a multi-frame model for video-level deepfake detection.

    This model processes multiple frames from a video and aggregates
    their features using the specified temporal mode.

    Args:
        backbone: CNN backbone architecture ('efficientnet_b0', 'resnet18', etc.)
        temporal_mode: Temporal aggregation method
            - 'mean': Average frame features (recommended for simplicity)
            - 'lstm': LSTM over frame features
            - 'gru': GRU over frame features
            - 'attention': Self-attention over frame features
        num_classes: Number of output classes
        hidden_size: Hidden size for LSTM/GRU/attention layers
        num_layers: Number of layers for LSTM/GRU
        pretrained: Whether to use pretrained ImageNet weights

    Returns:
        MultiFrameModel that accepts (B, T, C, H, W) input and outputs (B, num_classes)
    """
    # Get the CNN backbone
    cnn = get_model(backbone, num_classes=hidden_size, pretrained=pretrained)

    return MultiFrameModel(
        cnn=cnn,
        hidden_size=hidden_size,
        num_classes=num_classes,
        temporal_mode=temporal_mode,
        num_layers=num_layers,
    )


class MultiFrameModel(nn.Module):
    """
    Multi-frame video classification model.

    Processes multiple frames through a CNN backbone and aggregates
    features using temporal modeling.
    """

    def __init__(
        self,
        cnn: nn.Module,
        hidden_size: int,
        num_classes: int,
        temporal_mode: str = "mean",
        num_layers: int = 1,
    ):
        super().__init__()

        self.cnn = cnn
        self.temporal_mode = temporal_mode

        # Get feature dimension from CNN
        if hasattr(cnn, "classifier"):
            # EfficientNet
            self.feature_dim = cnn.classifier[1].in_features
            # Remove original classifier
            self.cnn.classifier = nn.Identity()
        elif hasattr(cnn, "fc"):
            # ResNet
            self.feature_dim = cnn.fc.in_features
            self.cnn.fc = nn.Identity()
        elif hasattr(cnn, "classifier") and hasattr(cnn.classifier, "[1]"):
            # MobileNet
            self.feature_dim = 1280
            self.cnn.classifier = nn.Identity()
        else:
            self.feature_dim = hidden_size

        self.feature_dim = hidden_size  # Use hidden_size as feature dim

        # Projection layer to ensure consistent feature dimension
        self.projection = nn.Linear(self.feature_dim, hidden_size)

        # Temporal aggregation layers
        if temporal_mode == "mean":
            self.temporal_agg = None  # Simple mean pooling

        elif temporal_mode == "lstm":
            self.temporal_agg = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
            # Output projection to match num_classes
            self.temporal_proj = nn.Linear(hidden_size * 2, hidden_size)

        elif temporal_mode == "gru":
            self.temporal_agg = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.temporal_proj = nn.Linear(hidden_size * 2, hidden_size)

        elif temporal_mode == "attention":
            self.temporal_agg = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True,
            )
            # Learnable query for video-level representation
            self.video_query = nn.Parameter(torch.randn(1, 1, hidden_size))
            self.temporal_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown temporal mode: {temporal_mode}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W) where B=batch, T=frames,
               C=channels, H=height, W=width

        Returns:
            Tensor of shape (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Reshape to process all frames through CNN
        x = x.view(B * T, C, H, W)

        # Extract features
        features = self.cnn(x)  # (B*T, feature_dim)

        # Project to hidden size
        features = self.projection(features)  # (B*T, hidden_size)

        # Reshape for temporal processing
        features = features.view(B, T, -1)  # (B, T, hidden_size)

        # Temporal aggregation
        if self.temporal_mode == "mean":
            # Mean pooling across time
            video_features = features.mean(dim=1)  # (B, hidden_size)

        elif self.temporal_mode == "lstm":
            # LSTM processing
            lstm_out, (h_n, c_n) = self.temporal_agg(features)
            # Use last hidden state from both directions
            video_features = self.temporal_proj(lstm_out[:, -1, :])

        elif self.temporal_mode == "gru":
            # GRU processing
            gru_out, h_n = self.temporal_agg(features)
            # Use last hidden state from both directions
            video_features = self.temporal_proj(gru_out[:, -1, :])

        elif self.temporal_mode == "attention":
            # Self-attention with learnable query
            B, T, _ = features.shape
            query = self.video_query.expand(B, -1, -1)  # (B, 1, hidden_size)
            attn_out, _ = self.temporal_agg(query, features, features)
            video_features = attn_out.squeeze(1)  # (B, hidden_size)

        # Classification
        output = self.classifier(video_features)

        return output
