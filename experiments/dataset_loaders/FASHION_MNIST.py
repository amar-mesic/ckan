import torch
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

def get_FASHION_MNIST(fraction, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    mnist_train = FashionMNIST(root='../data', train=True, download=True, transform=transform)
    mnist_test = FashionMNIST(root='../data', train=False, download=True, transform=transform)

    # Define a function to create a subset of the dataset
    def get_subset(dataset, fraction):
        subset_size = int(len(dataset) * fraction)
        indices = torch.randperm(len(dataset))[:subset_size]
        return Subset(dataset, indices)

    # Create subsets with only X% of the data
    # This speeds up training immensely
    train_val_dataset = get_subset(mnist_train, fraction)
    test_dataset = get_subset(mnist_test, fraction)

    # DataLoader
    train_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_val_dataset,
        test_dataset,
        train_loader,
        test_loader
    )