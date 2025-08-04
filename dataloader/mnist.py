import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MnistDataset(Dataset):
    def __init__(
        self,
        mnist_dataset: datasets.MNIST,
        crop_noise: bool = False,
        crop_size: int = 12,
        shuffle_pairs: bool = False,
    ) -> None:
        """
        MNIST dataset wrapper for ProbAI tasks.
        :param mnist_dataset: The MNIST dataset to wrap.
        :param crop_noise: If True, crops a square of pixels and sets them to 1 (white).
        :param crop_size: Size of the cropped square.
        :param shuffle_pairs: If True, shuffles the pairs of images.
        """
        self.mnist_dataset = mnist_dataset
        self.crop_noise = crop_noise
        self.shuffle_pairs = shuffle_pairs
        if shuffle_pairs:
            self.shuffle_idx = torch.randperm(len(mnist_dataset))

        # Can remain unchanged, but feel free to experiment with this parameter
        self.crop_size = crop_size

        self.img_dim0 = mnist_dataset.data.shape[1]
        self.img_dim1 = mnist_dataset.data.shape[2]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."
        :return: Number of samples in the dataset.
        """
        # We don't care about the label so we only extact [0] from the wrapped dataset
        return self.mnist_dataset.__len__()

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a pair of images from the dataset.
        :param idx: Index of the sample to retrieve.
        :return: A tuple containing two images (x0, x1).
        """
        # We don't care about the label so we only extact [0] from the wrapped dataset
        x1 = self.mnist_dataset.__getitem__(idx)[0].detach()
        if self.crop_noise:
            if self.shuffle_pairs:
                idx = self.shuffle_idx[idx]
            x0 = self.mnist_dataset.__getitem__(idx)[0].detach().clone()

            # Sample a square of pixels and set them to 1 (white)
            idx0 = torch.randint(0, self.img_dim0 - self.crop_size, (1,))[0]
            idx1 = torch.randint(0, self.img_dim1 - self.crop_size, (1,))[0]
            x0[0, idx0 : idx0 + self.crop_size, idx1 : idx1 + self.crop_size] = 1
        else:
            x0 = torch.randn(1, self.img_dim0, self.img_dim0)
        return x0, x1

    def reshuffle_pairs(self) -> None:
        """Rearrange the pairs of images in the dataset."""
        """
        Rearrange the pairs of images in the dataset.
        This is useful if you want to change the pairs after each epoch.
        :return: None
        """
        # You can call this between epochs to rearange the pairs
        # Is probably not that influential/important given the size of the dataset and how few epochs we need
        if self.shuffle_pairs:
            self.shuffle_idx = torch.randperm(len(self.mnist_dataset))


def test_dataloader(path: Path):
    """
    Test the data loader.
    :param path: Path to the dataset.
    :return: None
    """
    # Load MNIST datsets, init ProbAI dataset objects and create train and test dataloaders
    batch_size = 10
    crop = False  # True if config 2 and 3
    shuffle = False  # True if config 2
    trainset = datasets.MNIST(
        path,
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )

    testset = datasets.MNIST(
        path,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        MnistDataset(trainset, crop_noise=crop, shuffle_pairs=shuffle),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        MnistDataset(testset, crop_noise=crop, shuffle_pairs=shuffle),
        batch_size=batch_size,
        drop_last=True,
    )
    print(f"Train set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    print(f"Train loader size: {len(train_loader)}")
    print(f"Test loader size: {len(test_loader)}")

    # Check if the dataloader works
    for x0, x1 in train_loader:
        print(f"x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        break
    for x0, x1 in test_loader:
        print(f"x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        break


def parse_args():
    parser = argparse.ArgumentParser(description="Test MNIST dataloader")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to save the MNIST dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    test_dataloader(data_path)
