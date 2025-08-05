"""
Trainer for Gaussian Prior using Conditional Flow Matching
This script defines a Trainer class that initializes the model, data, optimizer,
and flow matcher for training a Gaussian Prior model on the MNIST dataset.
It includes methods for training, evaluation, and saving checkpoints.
This code is inspired from an example of TorchCFM library for CIFAR dataset.
https://github.com/atong01/conditional-flow-matching/tree/1.0.5/examples/cifar10
"""

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
from tqdm import tqdm, trange

import wandb
from dataloader.mnist import MnistDataset


def warmup_lr(step: int, warmup: int) -> float:
    return min(step, warmup) / warmup


class Trainer:
    def __init__(self, params: DictConfig) -> None:
        """
        Initializes the Trainer with the given parameters.
        Sets up the device, data loaders, model, optimizer, and flow matcher.
        Loads a checkpoint if available.
        """
        self.params = params
        self.device = params.main.device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # WandB initialization
        if params.main.use_wandb:
            wandb.init(
                project=params.main.wandb_project,
                name=params.main.name,
                config=dict(params),
            )

        self.save_dir = Path(params.main.save_dir) / params.main.name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_data()
        self._init_model()
        self._init_optimizer()
        self._init_flow_matcher()

        self.start_epoch = 0
        self._load_checkpoint_if_available()

        self.mse_metric = torch.nn.MSELoss()

    def _init_data(self) -> None:
        """
        Initializes the MNIST dataset and data loaders.
        Applies transformations to the images and sets up training and test datasets.
        """
        print("Initializing data...")
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Load MNIST dataset train and test sets
        train_raw = datasets.MNIST(
            self.params.data.path, train=True, download=False, transform=transform
        )
        test_raw = datasets.MNIST(
            self.params.data.path, train=False, download=False, transform=transform
        )

        self.train_dataset = MnistDataset(
            train_raw,
            crop_noise=self.params.data.crop_noise,
            shuffle_pairs=self.params.data.shuffle_pairs,
        )

        self.test_dataset = MnistDataset(
            test_raw,
            crop_noise=self.params.data.crop_noise,
            shuffle_pairs=self.params.data.shuffle_pairs,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.params.data.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.params.data.num_workers,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.params.data.batch_size,
            drop_last=True,
            num_workers=self.params.data.num_workers,
        )

        print(f"Train set size: {len(self.train_dataset)}")
        print(f"Test set size: {len(self.test_dataset)}")
        self.data_image, _ = self.train_dataset.__getitem__(0)
        print(f"Data image shape: {self.data_image.shape}")

    def _init_model(self) -> None:
        """
        Initializes the UNet model wrapped in a NeuralODE for velocity field estimation.
        Configures the model parameters based on the provided configuration.
        """

        # We use a UNet to model the velocity field u_t(x, t)
        print("Initializing model...")
        self.velocity_model = UNetModelWrapper(
            dim=tuple(self.data_image.shape),
            num_res_blocks=self.params.model.num_res_blocks,
            num_channels=self.params.model.num_channels,
            num_heads=self.params.model.num_heads,
            num_head_channels=self.params.model.num_head_channels,
            attention_resolutions=self.params.model.attention_resolutions,
            dropout=self.params.model.dropout,
        ).to(self.device)

        # ODE that defines the flow
        self.velocity_ode = NeuralODE(
            self.velocity_model, sensitivity="adjoint", solver=self.params.ode.solver
        )

        model_size = sum(p.numel() for p in self.velocity_ode.parameters())
        print(f"Model size: {model_size / (1024**2):.2f}M parameters")

    def _init_optimizer(self) -> None:
        """
        Initializes the optimizer and learning rate scheduler for the velocity model.
        Uses Adam optimizer with a learning rate and warmup steps defined in the parameters.
        """
        # We use Adam optimizer with a learning rate scheduler
        print("Initializing optimizer...")
        self.optimizer = torch.optim.Adam(
            self.velocity_model.parameters(), lr=self.params.optimizer.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_lr(step, self.params.optimizer.warmup_steps),
        )

    def _init_flow_matcher(self) -> None:
        """
        Initializes the flow matcher.
        """

        # Here we define the flow matcher, which determines the trajectories for
        # the pairs (x0, x1). This is directly related to the conditional z and q(z)
        sigma = 0.0
        matcher_type = self.params.flow.matcher
        if matcher_type == "exact":
            self.flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma)
        elif matcher_type == "target":
            self.flow_matcher = TargetConditionalFlowMatcher(sigma)
        elif matcher_type == "conditional":
            self.flow_matcher = ConditionalFlowMatcher(sigma)
        else:
            raise ValueError(f"Unknown flow matcher: {matcher_type}")

    def _load_checkpoint_if_available(self) -> None:
        """
        Loads a checkpoint if available.
        If a checkpoint exists, it loads the model state, optimizer state, and scheduler state.
        Sets the starting epoch for training based on the loaded checkpoint.
        """
        print("Loading checkpoint if available...")
        checkpoint_path = self.params.main.checkpoint_path
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.velocity_model.load_state_dict(checkpoint["velocity_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1
            print(
                f"Loaded checkpoint from {checkpoint_path} at epoch {self.start_epoch}"
            )
        else:
            print("No checkpoint found, starting fresh.")

    def save_checkpoint(self, epoch: int) -> None:
        """
        Saves the current state of the model, optimizer, and scheduler to a checkpoint file.
        The checkpoint includes the model state dictionary, optimizer state dictionary,
        scheduler state dictionary, and the current epoch number.
        :param epoch: The current epoch number to save in the checkpoint.
        """
        print(f"Saving checkpoint at epoch {epoch + 1}...")
        save_dict = {
            "velocity_model_state_dict": self.velocity_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
        }
        torch.save(save_dict, self.save_dir / f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(save_dict, self.save_dir / "latest.pt")
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def evaluate(self, epoch: int) -> None:
        """
        Evaluates the model on the test dataset.
        Computes the mean squared error (MSE) between predicted and actual images.
        Saves sample images of predictions, x0, and x1 for visual inspection.
        :param epoch: The current epoch number for logging purposes.
        """
        print(f"Evaluating model at epoch {epoch + 1}...")
        self.velocity_model.eval()
        self.velocity_ode.eval()
        total_mse = 0.0
        batches = 0

        with torch.inference_mode():
            images_save_dir = self.save_dir / f"images_epoch_{epoch + 1}"
            images_save_dir.mkdir(parents=True, exist_ok=True)
            for j, (x0, x1) in enumerate(self.test_loader):
                # Here we sample from the source distribution p(x0) and use the ODE
                # to integrate a path and generate x1.
                x0, x1 = x0.to(self.device), x1.to(self.device)
                traj = self.velocity_ode.trajectory(
                    x0,
                    torch.linspace(0, 1, self.params.ode.ode_num_steps).to(self.device),
                )
                x1_pred = traj[-1, :].view_as(x1).clip(-1, 1)
                total_mse += self.mse_metric(x1_pred, x1).item()
                batches += 1

                if j == 0:
                    # Save sample images for visual inspection
                    torchvision.utils.save_image(
                        (x1_pred + 1) / 2, images_save_dir / f"image_{j + 1}.png"
                    )
                    torchvision.utils.save_image(
                        (x0 + 1) / 2, images_save_dir / f"x0_{j + 1}.png"
                    )
                    torchvision.utils.save_image(
                        (x1 + 1) / 2, images_save_dir / f"x1_{j + 1}.png"
                    )

                    if self.params.main.use_wandb:
                        wandb.log(
                            {
                                "images/pred": [
                                    wandb.Image(images_save_dir / f"image_{j + 1}.png")
                                ],
                                "images/x0": [
                                    wandb.Image(images_save_dir / f"x0_{j + 1}.png")
                                ],
                                "images/x1": [
                                    wandb.Image(images_save_dir / f"x1_{j + 1}.png")
                                ],
                            }
                        )
                    break

        avg_mse = total_mse / batches
        print(f"Evaluation MSE at epoch {epoch + 1}: {avg_mse:.6f}")
        if self.params.main.use_wandb:
            wandb.log({"eval/mse": avg_mse, "epoch": epoch + 1})

    def train(self) -> None:
        """
        Trains the model for a specified number of epochs.
        For each epoch, it iterates over the training data, computes the loss,
        performs backpropagation, and updates the model parameters.
        Logs the average loss and saves checkpoints at specified intervals.
        """
        print("Starting training...")
        with trange(
            self.start_epoch, self.params.main.epochs, dynamic_ncols=True
        ) as pbar:
            for epoch in pbar:
                self.velocity_model.train()
                self.velocity_ode.train()
                average_epoch_loss = 0.0
                pbar.set_description(f"Epoch {epoch + 1}/{self.params.main.epochs}")
                pbar2 = tqdm(
                    self.train_loader, desc="Training", leave=False, dynamic_ncols=True
                )

                for i, (x0, x1) in enumerate(pbar2):
                    self.optimizer.zero_grad()
                    # Sample from the source and target distributions
                    x0, x1 = x0.to(self.device), x1.to(self.device)
                    # Sample the time t, velocity field u_t, and the conditional flow
                    t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                        x0, x1
                    )
                    # Compute the velocity field u_t(x_t, t)
                    vt = self.velocity_model(t, xt)
                    # Compute the loss as the mean squared error between v_t and u_t
                    loss = torch.mean((vt - ut) ** 2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.velocity_model.parameters(),
                        self.params.optimizer.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    pbar2.set_postfix({"loss": loss.item()})
                    average_epoch_loss += loss.item()

                    if self.params.main.use_wandb:
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "step": i + epoch * len(self.train_loader),
                            }
                        )

                average_epoch_loss /= len(self.train_loader)
                pbar.set_postfix({"avg_loss": average_epoch_loss})
                print(f"Epoch {epoch + 1} average loss: {average_epoch_loss:.4f}")

                if self.params.main.use_wandb:
                    wandb.log(
                        {"train/avg_loss": average_epoch_loss, "epoch": epoch + 1}
                    )

                if (epoch + 1) % self.params.main.save_freq == 0:
                    self.save_checkpoint(epoch)
                    self.evaluate(epoch)

        if self.params.main.use_wandb:
            wandb.finish()


@hydra.main(version_base=None, config_path="./config", config_name="gaussian_prior")
def main(params: DictConfig) -> None:
    """
    Main function to run the Trainer.
    Initializes the Trainer with the provided parameters and starts the training process.
    """
    print("Starting Gaussian Prior Trainer...")
    print(f"Configuration: {params}")
    trainer = Trainer(params)
    trainer.train()


if __name__ == "__main__":
    main()
