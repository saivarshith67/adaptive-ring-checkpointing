import torch
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


def get_model(name, num_classes=100):

    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(1280, num_classes)

    elif name == "efficientnet_b0":
        model = get_efficientnet_binary(pretrained=True)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model
