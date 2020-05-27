
"""Trainer class."""

import collections
import copy
import json
import logging
import pathlib
import time

import tqdm

import torch
from torch import optim
import tensorboardX as tb

import gqnlib


class Trainer:
    """Trainer class for neural process.

    Args:
        model (gqnlib.GenerativeQueryNetwork): GQN model.
        hparams (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model: gqnlib.GenerativeQueryNetwork, hparams: dict):
        # Params
        self.model = model
        self.hparams = hparams

        # Attributes
        self.logdir = None
        self.logger = None
        self.writer = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.device = None

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
        self.logdir.mkdir(parents=True)

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

    def load_dataloader(self) -> None:
        """Loads data loader for training and test."""

        self.logger.info("Load dataset")

        self.train_loader = torch.utils.data.DataLoader(
            gqnlib.SceneDataset(self.hparams["train_dir"]),
            shuffle=True, batch_size=1)

        self.test_loader = torch.utils.data.DataLoader(
            gqnlib.SceneDataset(self.hparams["test_dir"]),
            shuffle=False, batch_size=1)

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

            # Forward
            self.optimizer.zero_grad()
            _tmp_loss_dict = self.model.loss_func(*data)
            loss = _tmp_loss_dict["loss"]

            # Backward
            loss.backward()
            self.optimizer.step()

            # Save loss
            for key, value in _tmp_loss_dict.items():
                loss_dict[key] += value.item()

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

        # Data
        self.load_dataloader()

        # Model to device
        if self.hparams["gpus"] is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.hparams['gpus']}")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters())

        # Run training
        self.logger.info("Start training")
        pbar = tqdm.trange(1, self.hparams["epochs"] + 1)
        postfix = {"train/loss": 0, "test/loss": 0}
        for epoch in pbar:
            # Training
            train_loss = self.train(epoch)
            postfix["train/loss"] = train_loss

            if epoch % self.hparams["log_save_interval"] == 0:
                # Calculate test loss
                test_loss = self.test(epoch)
                postfix["test/loss"] = test_loss
                self.save_checkpoint(epoch, test_loss)
                self.save_plot(epoch)

            # Update postfix
            pbar.set_postfix(postfix)

        # Post process
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