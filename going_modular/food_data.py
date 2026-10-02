"""
Food-101 dataloaders.

Lives in a .py file so DataLoader worker processes can re-import it cleanly on
Windows — workers re-import the defining module, and a notebook cell is not a
safely importable module, which is why num_workers > 0 hangs in Jupyter.

    from Going_modular.food_data import create_dataloaders
    train_dl, test_dl, class_names = create_dataloaders(batch_size=64,
                                                        num_workers=4,
                                                        augment=True)
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(root="data", batch_size=64, num_workers=4, img_size=224,
                       augment=True):
    """Returns (train_dataloader, test_dataloader, class_names) for full Food-101.

    augment=True applies the DeiT-style train recipe. A from-scratch ViT has no
    built-in visual priors, so the augmentation supplies the invariances a CNN
    gets for free from its architecture — it is doing real work here, not just
    regularising. Expect early losses to look WORSE than an unaugmented run.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])

    # Test transform stays deterministic — never augment what you measure on.
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

    train_data = datasets.Food101(root=root, split="train",
                                  transform=train_transform, download=False)
    test_data = datasets.Food101(root=root, split="test",
                                 transform=test_transform, download=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=num_workers > 0)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=num_workers > 0)

    print(f"[INFO] {len(train_data)} train / {len(test_data)} test images, "
          f"{len(train_data.classes)} classes, augment={augment}")
    return train_dataloader, test_dataloader, train_data.classes
