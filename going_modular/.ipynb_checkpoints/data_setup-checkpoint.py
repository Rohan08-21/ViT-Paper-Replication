"""
Contains functionality for creating PyTorch DataLoaders for image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 4

def create_datalaoders(
    train_dir,
    test_dir,
    transform: transform.Compose,
    batch_size,
    num_workers=NUM_WORKERS
):
    """
    Creating training and testing Dataloaders.

    Takes in a training directory and testing directory paths and turns them into
    PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    train_dir : Path to training directory
    test_dir : Patah to testing directory
    transform: torchvision transforms to perform on training and testing data
    batch_size: Number of samples per batch in each of the Dataloaders
    num_workers: An integer of number of workers per Dataloader

    Returns:
    A Tuple of train_dataloader, test_loader, class_names.
    Where class names is a list of target(labels) classes.

    """
    # Use ImageFolder to create dataset
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    # Get class names 
    class_names = train_data.classes

    # Turn our images into Dataloader
    train_dataloader = DataLoader(train_data,
                                 batch_size=24,
                                 num_workers=4,
                                 shuffle=True,
                                 pin_memory=True)

    test_dataloader = DataLoader(test_data,
                                batch_size=24,
                                num_workers=4,
                                shuffle=False,
                                pin_memory=True)

    return train_dataloader, test_dataloader, class_names
