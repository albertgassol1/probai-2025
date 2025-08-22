import argparse
from pathlib import Path
from typing import Tuple

import torch
import torchvision.utils as vutils
from omegaconf import DictConfig, OmegaConf
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms

from dataloader.mnist import MnistDataset


def load_model(
    checkpoint_path: str, device: str, image_shape: Tuple[int, int], params: DictConfig
) -> Tuple[UNetModelWrapper, NeuralODE]:
    """Load trained UNet wrapped in NeuralODE.
    :param checkpoint_path: Path to the model checkpoint.
    :param device: Device to load the model on.
    :param image_shape: Shape of the input images.
    :param params: Configuration parameters.
    :return: Loaded model and ODE.
    """
    model = UNetModelWrapper(
        dim=tuple(image_shape),
        num_res_blocks=params.model.num_res_blocks,
        num_channels=params.model.num_channels,
        num_heads=params.model.num_heads,
        num_head_channels=params.model.num_head_channels,
        attention_resolutions=params.model.attention_resolutions,
        dropout=params.model.dropout,
    ).to(device)

    ode = NeuralODE(model, sensitivity="adjoint", solver=params.ode.solver)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["velocity_model_state_dict"])
    model.eval(), ode.eval()
    return model, ode


@torch.inference_mode()
def main(args, params: DictConfig) -> None:
    """
    Main function to generate random digits.
    :param args: Command line arguments.
    :param params: Configuration parameters.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    test_raw = datasets.MNIST(
        params.data.path, train=False, download=False, transform=transform
    )
    test_dataset = MnistDataset(
        test_raw,
        crop_noise=params.data.crop_noise,
        shuffle_pairs=params.data.shuffle_pairs,
    )

    # Load model
    sample_img, _ = test_dataset[0]
    _, velocity_ode = load_model(args.model_path, device, sample_img.shape, params)

    # 1. Generate 100 digits source vs target
    x0_list, x1_pred_list = [], []
    for i in range(100):
        x0, _ = test_dataset[i]
        x0 = x0.unsqueeze(0).to(device)
        traj = velocity_ode.trajectory(
            x0, torch.linspace(0, 1, params.ode.ode_num_steps).to(device)
        )
        x1_pred = traj[-1].cpu().clamp(-1, 1)
        x0_list.append(x0.cpu())
        x1_pred_list.append(x1_pred)

    x0_grid = vutils.make_grid(
        torch.cat(x0_list), nrow=10, normalize=True, value_range=(-1, 1)
    )
    x1_grid = vutils.make_grid(
        torch.cat(x1_pred_list), nrow=10, normalize=True, value_range=(-1, 1)
    )
    vutils.save_image(x0_grid, save_dir / "source_digits.png")
    vutils.save_image(x1_grid, save_dir / "target_digits.png")

    # 2. Trajectories for 4 digits across 10 timesteps
    digits = [test_dataset[i][0].unsqueeze(0).to(device) for i in range(4)]
    x0_batch = torch.cat(digits, dim=0)

    ts = torch.linspace(0, 1, 10).to(device)
    traj = velocity_ode.trajectory(x0_batch, ts)

    # Save one grid per timestep (2x2 layout for 4 digits)
    for j, t in enumerate(ts):
        grid = vutils.make_grid(
            traj[j].cpu(),
            nrow=2,
            normalize=True,
            value_range=(-1, 1),
        )
        vutils.save_image(grid, save_dir / f"trajectory_timestep_{t}.png")

    print(f"Generated images saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save outputs"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/gaussian_prior.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    params = OmegaConf.load(args.config_path)
    main(args, params)
