from pathlib import Path

import hydra
import torch
import torchvision.utils
from omegaconf import DictConfig
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange

from dataloader.mnist import MnistDataset


def warmup_lr(step: int, warmup: int) -> float:
    """
    Warmup learning rate function.
    :param step: Current training step.
    :param warmup: Total warmup steps.
    :return: Warmup learning rate.
    """
    return min(step, warmup) / warmup


@hydra.main(version_base=None, config_path="../config", config_name="gaussian_prior")
def main(params: DictConfig) -> None:
    """
    Main entry point for the Gaussian prior training script.
    :param params: Configuration parameters for the training.
    """
    device = params.main.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_dataset = MnistDataset(
        datasets.MNIST(
            params.data.path,
            train=True,
            download=False,
            transform=mnist_transform,
        ),
        crop_noise=params.data.crop_noise,
        shuffle_pairs=params.data.shuffle_pairs,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=params.data.num_workers,
    )

    test_dataset = MnistDataset(
        datasets.MNIST(
            params.data.path,
            train=False,
            download=False,
            transform=mnist_transform,
        ),
        crop_noise=params.data.crop_noise,
        shuffle_pairs=params.data.shuffle_pairs,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params.data.batch_size,
        drop_last=True,
        num_workers=params.data.num_workers,
    )

    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Initialize the model
    velocity_model = UNetModelWrapper(
        dim=params.model.dim,
        num_res_blocks=params.model.num_res_blocks,
        num_channels=params.model.num_channels,
        channel_mult=params.model.channel_mult,
        num_heads=params.model.num_heads,
        num_head_channels=params.model.num_head_channels,
        attention_resolutions=params.model.attention_resolutions,
        dropout=params.model.dropout,
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(velocity_model.parameters(), lr=params.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_lr(step, params.optimizer.warmup_steps)
    )

    # ODE model
    velocity_ode = NeuralODE(
        velocity_model, sensitivity="adjoint", solver=params.model.solver
    )
    model_size = sum(param.data.nelement() for param in velocity_ode.parameters())
    print(f"Model size: {model_size / (1024**2):.2f}M parameters")

    # Flow matcher
    sigma = 0.0
    if params.flow.matcher == "exact":
        flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif params.flow.matcher == "target":
        flow_matcher = TargetConditionalFlowMatcher(sigma=sigma)
    elif params.flow.matcher == "conditional":
        flow_matcher = ConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError(f"Unknown flow matcher: {params.flow.matcher}")

    # Save directory
    save_dir = Path(params.main.save_dir) / params.main.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    with trange(params.main.epochs, dynamic_ncols=True) as pbar:
        for epoch in range(params.main.epochs):
            velocity_model.train()
            velocity_ode.train()
            for batch_idx, (x0, x1) in enumerate(train_dataloader):
                optimizer.zero_grad()
                x0, x1 = x0.to(device), x1.to(device)
                t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
                vt = velocity_model(t, xt)
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    velocity_model.parameters(), params.optimizer.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)
            if (epoch + 1) % params.main.save_freq == 0:
                save_dict = {
                    "velocity_model_state_dict": velocity_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                }
                torch.save(save_dict, save_dir / f"checkpoint_epoch_{epoch + 1}.pt")
                torch.save(save_dict, "latest.pt")
                print(f"Checkpoint saved at epoch {epoch + 1}")
                # Evaluate on test set and save images
                velocity_model.eval()
                velocity_ode.eval()
                with torch.inference_mode():
                    images_save_dir = save_dir / f"images_epoch_{epoch + 1}"
                    images_save_dir.mkdir(parents=True, exist_ok=True)
                    for j, (x0, x1) in enumerate(test_dataloader):
                        x0, x1 = x0.to(device), x1.to(device)
                        traj = velocity_ode.trajectory(
                            x0,
                            torch.linspace(0, 1, params.model.ode_num_steps).to(device),
                        )
                        traj = traj[-1, :].view(-1, 3, 32, 32).clip(-1, 1)
                        traj = (traj + 1) / 2  # Normalize to [0, 1]
                        torchvision.utils.save_image(
                            traj, images_save_dir / f"image_{j + 1}.png"
                        )
                        torchvision.utils.save_image(
                            x0, images_save_dir / f"x0_{j + 1}.png"
                        )


if __name__ == "__main__":
    main()
