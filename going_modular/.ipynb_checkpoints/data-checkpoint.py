from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(root="data", batch_size=64, num_workers=4, img_size=224):
    """Returns (train_dataloader, test_dataloader, class_names) for full Food-101."""
    vit_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    train_data = datasets.Food101(root=root, split="train",
                                  transform=vit_transform, download=False)
    test_data = datasets.Food101(root=root, split="test",
                                 transform=vit_transform, download=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=num_workers > 0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=num_workers > 0)

    return train_dataloader, test_dataloader, train_data.classes
