from torchvision import transforms
from torchvision.datasets import CIFAR100


def get_cifar100_trainset(datadir_root, transform=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
    ])

    return CIFAR100(
        root=datadir_root,
        train=True,
        download=True,
        transform=transform_train if transform else None,
    )


def get_cifar_100_testset(datadir_root, transform=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
    ])

    return CIFAR100(
        root=datadir_root,
        train=False,
        download=True,
        transform=transform_test if transform else None,
    )