import argparse
from pathlib import Path

from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Download MNIST dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to save the MNIST dataset",
    )
    return parser.parse_args()


def download_mnist(data_path: Path):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    testset = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return trainset, testset


if __name__ == "__main__":
    args = parse_args()
    path = Path(args.data_path)
    path.mkdir(parents=True, exist_ok=True)
    trainset, testset = download_mnist(path)
    print(f"MNIST dataset downloaded to {path}")
    print(f"Train set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
