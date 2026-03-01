import torchvision
import torchvision.transforms as transforms


def get_cifar100_dataset(root="./data", train=True):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )

    return dataset