import torch
import torchvision.models as models


def get_model(name, num_classes=100):

    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, num_classes)

    elif name == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(1280, num_classes)

    else:
        raise ValueError("Unknown model")

    return model