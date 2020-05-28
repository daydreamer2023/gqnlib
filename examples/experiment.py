
"""Trainer class."""

import collections
import copy
import json
import logging
import math
import pathlib
import time

import tqdm

import torch
from torch import nn, optim, distributed
import tensorboardX as tb

import gqnlib


class Trainer:
    """Trainer class for neural process.

    **Notes**

    `hparams` should include the following keys.

    * logdir (str): Path to log direcotry. This is updated to `logdir/<date>`.
    * train_dir (str): Path to training data.
    * test_dir (str): Path to test data.
    * batch_size (int): Batch size.
    * max_steps (int): Number of max iteration steps.
    * log_save_interval (int): Number of interval epochs to save checkpoints.
    * gpus (str): Comma separated list of GPU IDs (ex. '0,1').

    Args:
        model (gqnlib.GenerativeQueryNetwork): GQN model.
        hparams (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model: gqnlib.GenerativeQueryNetwork, hparams: dict):
        # Params
        self.model = model
        self.hparams = hparams

        # Attributes
        self.logdir = pathlib.Path()
        self.logger = None
        self.writer = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.device = None
        self.global_steps = 0
        self.max_steps = 0

    def check_logdir(self) -> None:
        """Checks log directory.

        This method specifies logdir and make the directory if it does not
        exist.
        """

        if "logdir" in self.hparams:
            logdir = pathlib.Path(self.hparams["logdir"])
        else:
            logdir = pathlib.Path("./logs/tmp/")

        self.logdir = logdir / time.strftime("%Y%m%d%H%M")
        self.logdir.mkdir(parents=True, exist_ok=True)

    def init_logger(self, save_file: bool = True) -> None:
        """Initalizes logger.

        Args:
            save_file (bool, optoinal): If `True`, save log file.
        """

        # Log file
        logpath = self.logdir / "training.log"

        # Initialize logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # Set stream handler (console)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh_fmt = logging.Formatter("%(asctime)s - %(module)s.%(funcName)s "
                                   "- %(levelname)s : %(message)s")
        sh.setFormatter(sh_fmt)
        logger.addHandler(sh)

        # Set file handler (log file)
        if save_file:
            fh = logging.FileHandler(filename=logpath)
            fh.setLevel(logging.DEBUG)
            fh_fmt = logging.Formatter("%(asctime)s - %(module)s.%(funcName)s "
                                       "- %(levelname)s : %(message)s")
            fh.setFormatter(fh_fmt)
            logger.addHandler(fh)

        self.logger = logger

    def init_writer(self) -> None:
        """Initializes tensorboard writer."""

        self.writer = tb.SummaryWriter(str(self.logdir))

    def load_dataloader(self, train_dir: str, test_dir: str, batch_size: int
                        ) -> None:
        """Loads data loader for training and test.

        Args:
            train_dir (str): Path to train directory.
            test_dir (str): Path to test directory.
            batch_size (int): Batch size.
        """

        self.logger.info("Load dataset")

        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self.train_loader = torch.utils.data.DataLoader(
            gqnlib.SceneDataset(train_dir), shuffle=True,
            batch_size=batch_size, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            gqnlib.SceneDataset(test_dir), shuffle=False,
            batch_size=batch_size, **kwargs)

    def train(self, epoch: int) -> float:
        """Trains model.

        Args:
            epoch (int): Current epoch.

        Returns:
            train_loss (float): Accumulated loss value.
        """

        # Logger for loss
        loss_dict = collections.defaultdict(float)

        # Run
        self.model.train()
        for data in self.train_loader:
            # Split data into context and query
            data = gqnlib.partition(*data)

            # Data to device
            data = (x.to(self.device) for x in data)

            # Pixel variance annealing
            var = next(self.sigma_scheduler) ** 2

            # Forward
            self.optimizer.zero_grad()
            _tmp_loss_dict = self.model.loss_func(*data, var)
            loss = _tmp_loss_dict["loss"]

            # Backward
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.global_steps += 1

            # Save loss
            for key, value in _tmp_loss_dict.items():
                loss_dict[key] += value.item()

            # Check step limit
            if self.global_steps >= self.max_steps:
                break

        # Summary
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)

        return loss_dict["loss"]

    def test(self, epoch: int) -> float:
        """Tests model.

        Args:
            epoch (int): Current epoch.

        Returns:
            test_loss (float): Accumulated loss per iteration.
        """

        # Logger for loss
        loss_dict = collections.defaultdict(float)

        # Run
        self.model.eval()
        for data in self.test_loader:
            with torch.no_grad():
                # Split data into context and query
                data = gqnlib.partition(*data)

                # Data to device
                data = (v.to(self.device) for v in data)
                _tmp_loss_dict = self.model.loss_func(*data)

            # Save loss
            for key, value in _tmp_loss_dict.items():
                loss_dict[key] += value.item()

        # Summary
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"test/{key}", value, epoch)

        return loss_dict["loss"]

    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Saves trained model and optimizer to checkpoint file.

        Args:
            epoch (int): Current epoch number.
            loss (float): Saved loss value.
        """

        # Log
        self.logger.debug(f"Eval loss (epoch={epoch}): {loss}")
        self.logger.debug("Save trained model")

        # Save model
        state_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
        }
        path = self.logdir / f"checkpoint_{epoch}.pt"
        torch.save(state_dict, path)

    def save_configs(self) -> None:
        """Saves setting including condig and args in json format."""

        config = copy.deepcopy(self.hparams)
        config["logdir"] = str(self.logdir)

        with (self.logdir / "config.json").open("w") as f:
            json.dump(config, f)

    def save_plot(self, epoch: int) -> None:
        """Plot and save a figure.

        Args:
            epoch (int): Number of epoch.
        """

        pass

    def quit(self) -> None:
        """Post process."""

        self.logger.info("No data loader is saved")
        self.writer.close()

    def _base_run(self) -> None:
        """Base running method."""

        self.logger.info("Start experiment")

        # Pop hyper parameters
        hparams = copy.deepcopy(self.hparams)
        train_dir = hparams.pop("train_dir", "./data/tmp/train")
        test_dir = hparams.pop("test_dir", "./data/tmp/test")
        batch_size = hparams.pop("batch_size", 1)
        self.max_steps = hparams.pop("steps", 10)
        log_save_interval = hparams.pop("log_save_interval", 5)
        gpus = hparams.pop("gpus", None)

        optimizer_params = hparams.pop("optimizer_params", {})
        lr_scheduler_params = hparams.pop("lr_scheduler_params", {})
        sigma_scheduler_params = hparams.pop(
            "sigma_scheduler_params",
            {"init": 2.0, "final": 0.7, "steps": 80000})

        # Device
        if gpus:
            device_ids = list(map(int, gpus.split(",")))
            self.device = torch.device(f"cuda:{device_ids[0]}")
        else:
            device_ids = []
            self.device = torch.device("cpu")

        # Data
        self.load_dataloader(train_dir, test_dir, batch_size, gpus)

        # Model
        self.model = self.model.to(self.device)

        if len(device_ids) > 1:
            # Distributed data parallel
            distributed.init_process_group()
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), **optimizer_params)

        # Annealing scheduler
        self.lr_scheduler = gqnlib.AnnealingStepLR(
            self.optimizer, **lr_scheduler_params)
        self.sigma_scheduler = gqnlib.Annealer(**sigma_scheduler_params)

        # Training iteration
        per_step = math.ceil(len(self.train_loader) / batch_size)
        max_epochs = math.ceil(self.max_steps / per_step)
        pbar = tqdm.trange(1, max_epochs + 1)
        postfix = {"steps": 0, "train/loss": 0, "test/loss": 0}
        self.global_steps = 0

        # Run training
        for epoch in pbar:
            # Training
            train_loss = self.train(epoch)
            postfix["train/loss"] = train_loss

            if epoch % log_save_interval == 0:
                # Calculate test loss
                test_loss = self.test(epoch)
                postfix["test/loss"] = test_loss
                self.save_checkpoint(epoch, test_loss)
                self.save_plot(epoch)

            # Update postfix
            postfix["steps"] = self.global_steps
            pbar.set_postfix(postfix)

            if self.global_steps >= self.max_steps:
                break

        # Post process
        test_loss = self.test(max_epochs)
        self.save_checkpoint(max_epochs, test_loss)
        self.save_configs()
        self.quit()

    def run(self) -> None:
        """Main run method."""

        # Settings
        self.check_logdir()
        self.init_logger()
        self.init_writer()

        self.logger.info("Start run")
        self.logger.info(f"Logdir: {self.logdir}")
        self.logger.info(f"Params: {self.hparams}")

        # Run
        try:
            self._base_run()
        except Exception as e:
            self.logger.exception(f"Run function error: {e}")

        self.logger.info("Finish run")
