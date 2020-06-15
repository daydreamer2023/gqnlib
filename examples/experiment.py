
"""Trainer class."""

import collections
import copy
import json
import logging
import pathlib
import time

import tqdm

import torch
from torch import nn, optim
import tensorboardX as tb

import gqnlib


class Trainer:
    """Trainer class for Generative Query Netowork.

    **Notes**

    `hparams` should include the following keys.

    * model (str): Model name.
    * logdir (str): Path to log direcotry. This is updated to `logdir/<date>`.
    * train_dir (str): Path to training data.
    * test_dir (str): Path to test data.
    * batch_size (int): Batch size.
    * max_steps (int): Number of max iteration steps.
    * test_interval (int): Number of interval epochs to test.
    * save_interval (int): Number of interval epochs to save checkpoints.
    * gpus (str): Comma separated list of GPU IDs (ex. '0,1').

    Args:
        model (gqnlib.GenerativeQueryNetwork): GQN model.
        hparams (dict): Dictionary of hyper-parameters.
    """

    def __init__(self, model: gqnlib.GenerativeQueryNetwork, hparams: dict):
        # Params
        self.model = model
        self.hparams = copy.deepcopy(hparams)

        # Attributes
        self.model_name = ""
        self.logdir = pathlib.Path()
        self.logger = None
        self.writer = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.device = None
        self.global_steps = 0
        self.max_steps = 0
        self.pbar = None
        self.postfix = {}
        self.test_interval = 10000

    def check_logdir(self) -> None:
        """Checks log directory.

        This method specifies logdir and make the directory if it does not
        exist.
        """

        logdir = self.hparams.get("logdir", "./logs/tmp/")
        self.logdir = pathlib.Path(logdir, time.strftime("%Y%m%d%H%M"))
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

        # Dataset specification
        if self.model_name == "sgqn":
            dataset = gqnlib.SlimDataset
        else:
            dataset = gqnlib.SceneDataset

        # Kwargs for dataset
        train_kwrags = {"root_dir": train_dir, "batch_size": batch_size}
        test_kwargs = {"root_dir": test_dir, "batch_size": batch_size}

        if self.model_name == "sgqn":
            vectorizer = gqnlib.WordVectorizer()
            train_kwrags.update({"vectorizer": vectorizer, "train": True})
            test_kwargs.update({"vectorizer": vectorizer, "train": False})

        # Params for GPU
        if torch.cuda.is_available():
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self.train_loader = torch.utils.data.DataLoader(
            dataset(**train_kwrags), shuffle=True, batch_size=1, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            dataset(**test_kwargs), shuffle=False, batch_size=1, **kwargs)

        self.logger.info(f"Train dataset size: {len(self.train_loader)}")
        self.logger.info(f"Test dataset size: {len(self.test_loader)}")

    def train(self) -> None:
        """Trains model."""

        # Partition method
        if self.model_name == "sgqn":
            partition = gqnlib.partition_slim
        else:
            partition = gqnlib.partition_scene

        for dataset in self.train_loader:
            for data in dataset:
                self.model.train()

                # Split data into context and query
                data = partition(*data)

                # Data to device
                data = (x.to(self.device) for x in data)

                # Pixel variance annealing
                var = next(self.sigma_scheduler) ** 2

                # Forward
                self.optimizer.zero_grad()
                loss_dict = self.model(*data, var)
                loss = loss_dict["loss"].mean()

                # Backward and update
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                # Progress bar update
                self.global_steps += 1
                self.pbar.update(1)

                self.postfix["train/loss"] = loss.item()
                self.pbar.set_postfix(self.postfix)

                # Summary
                for key, value in loss_dict.items():
                    self.writer.add_scalar(
                        f"train/{key}", value.mean(), self.global_steps)

                # Test
                if self.global_steps % self.test_interval == 0:
                    self.test()

                # Save checkpoint
                if self.global_steps % self.save_interval == 0:
                    self.save_checkpoint()

                # Check step limit
                if self.global_steps >= self.max_steps:
                    break

    def test(self) -> None:
        """Tests model."""

        # Partition method
        if self.model_name == "sgqn":
            partition = gqnlib.partition_slim
        else:
            partition = gqnlib.partition_scene

        # Logger for loss
        loss_logger = collections.defaultdict(float)
        count = 0

        # Run
        self.model.eval()
        for dataset in self.test_loader:
            for data in dataset:
                with torch.no_grad():
                    # Split data into context and query
                    data = partition(*data)

                    # Data to device
                    data = (v.to(self.device) for v in data)
                    loss_dict = self.model(*data)
                    loss = loss_dict["loss"]

                # Update progress bar
                self.postfix["test/loss"] = loss.mean().item()
                self.pbar.set_postfix(self.postfix)

                # Save loss
                count += loss.size(0)
                for key, value in loss_dict.items():
                    loss_logger[key] = value.sum().item()

        # Summary
        for key, value in loss_logger.items():
            self.writer.add_scalar(
                f"test/{key}", value / count, self.global_steps)

        self.logger.debug(
            f"Eval loss (steps={self.global_steps}): {loss_logger}")

    def save_checkpoint(self) -> None:
        """Saves trained model and optimizer to checkpoint file.

        Args:
            loss (float): Saved loss value.
        """

        # Log
        self.logger.debug("Save trained model")

        # Remove unnecessary prefix from state dict keys
        model_state_dict = {}
        for k, v in self.model.state_dict().items():
            model_state_dict[k.replace("module.", "")] = v

        optimizer_state_dict = {}
        for k, v in self.optimizer.state_dict().items():
            optimizer_state_dict[k.replace("module.", "")] = v

        # Save model
        state_dict = {
            "steps": self.global_steps,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        path = self.logdir / f"checkpoint_{self.global_steps}.pt"
        torch.save(state_dict, path)

    def save_configs(self) -> None:
        """Saves setting including config and args in json format."""

        self.logger.debug("Save configs")

        config = copy.deepcopy(self.hparams)
        config["logdir"] = str(self.logdir)

        with (self.logdir / "config.json").open("w") as f:
            json.dump(config, f)

    def quit(self) -> None:
        """Post process."""

        self.logger.info("Quit base run method")
        self.writer.close()

    def _base_run(self) -> None:
        """Base running method."""

        self.logger.info("Start experiment")

        # Pop hyper parameters
        model_name = self.hparams.get("model", "gqn")
        train_dir = self.hparams.get("train_dir", "./data/tmp/train")
        test_dir = self.hparams.get("test_dir", "./data/tmp/test")
        batch_size = self.hparams.get("batch_size", 1)
        max_steps = self.hparams.get("steps", 10)
        test_interval = self.hparams.get("test_interval", 5)
        save_interval = self.hparams.get("save_interval", 5)
        gpus = self.hparams.get("gpus", None)

        optimizer_params = self.hparams.get("optimizer_params", {})
        lr_scheduler_params = self.hparams.get("lr_scheduler_params", {})
        sigma_scheduler_params = self.hparams.get(
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
        self.model_name = model_name
        self.load_dataloader(train_dir, test_dir, batch_size)

        # Model
        self.model = self.model.to(self.device)

        if len(device_ids) > 1:
            # Data parallel
            self.model = nn.DataParallel(self.model, device_ids)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), **optimizer_params)

        # Annealing scheduler
        self.lr_scheduler = gqnlib.AnnealingStepLR(
            self.optimizer, **lr_scheduler_params)
        self.sigma_scheduler = gqnlib.Annealer(**sigma_scheduler_params)

        # Progress bar
        self.pbar = tqdm.tqdm(total=max_steps)
        self.global_steps = 0
        self.max_steps = max_steps
        self.postfix = {"train/loss": 0, "test/loss": 0}

        # Intervals
        self.test_interval = test_interval
        self.save_interval = save_interval

        # Run training
        while self.global_steps < self.max_steps:
            self.train()

        self.pbar.close()
        self.logger.info("Finish training")

        # Post process
        self.save_checkpoint()
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
