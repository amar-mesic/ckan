import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import deeplake
import numpy as np

deeplake_API_token = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJpZCI6InJlaW5vdXR2b3MiLCJhcGlfa2V5IjoiY2d3cUI4cU5vVlVLcUtlOUZBQ3RMbjFnSktMaVdQR2hTS2ZvcFRDTkRMSDl6In0."


# TODO: fix s.t. we can compute min max
class DeepLakeTransformedDataset(Dataset):
    def __init__(self, deeplake_dataset, transform=None):
        self.deeplake_dataset = deeplake_dataset
        self.transform = transform

    def __len__(self):
        return len(self.deeplake_dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        item = self.deeplake_dataset[idx]
        image = item['image']
        label = item['label']

        image = torch.as_tensor(np.array(image), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_TINY_IMAGENET(fraction, batch_size):
    # Load the dataset directly from DeepLake
    ds = deeplake.load("hub://activeloop/tiny-imagenet-train", token=deeplake_API_token)
    ds_test = deeplake.load("hub://activeloop/tiny-imagenet-test", token=deeplake_API_token)

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Assume input is tensor and needs conversion to PIL Image
        transforms.Resize((64, 64)),  # Adjust size if necessary
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = DeepLakeTransformedDataset(ds, transform=transform)
    test_dataset = DeepLakeTransformedDataset(ds_test, transform=transform)

    def get_subset(dataset, fraction):
        subset_size = int(len(dataset) * fraction)
        indices = torch.randperm(len(dataset))[:subset_size].tolist()
        return Subset(dataset, indices)

    train_val_dataset = get_subset(train_dataset, fraction)
    test_dataset = get_subset(test_dataset, fraction)

    train_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_val_dataset, test_dataset, train_loader, test_loader
