import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def getDataloaders(config):

    # Define the transformations (if any)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
    ])

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = int(config["train_val_split"] * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # 20% for validation

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader
