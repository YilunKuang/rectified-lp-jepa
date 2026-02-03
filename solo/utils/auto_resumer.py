from argparse import ArgumentParser
import json
import os
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

from omegaconf import DictConfig
from solo.utils.misc import omegaconf_select

Checkpoint = namedtuple("Checkpoint", ["creation_time", "args", "checkpoint"])


class AutoResumer:
    SHOULD_MATCH = [
        "name",
        "backbone",
        "method",
        "data.dataset",
        "max_epochs",
        "seed",
        "optimizer.name",
        "optimizer.batch_size",
        "optimizer.lr",
        "optimizer.weight_decay",
        "method_kwargs", # this would be a dictionary of all the method kwargs
        "wandb.project",
        "wandb.entity",
        "pretrained_feature_extractor",
    ] # wow this is major we need to add method kwargs here too or else it would be a major issue!!

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = Path("trained_models"),
        max_hours: int = 36,
    ):
        """Autoresumer object that automatically tries to find a checkpoint
        that is as old as max_time.

        Args:
            checkpoint_dir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            max_hours (int): maximum elapsed hours to consider checkpoint as valid.
        """

        self.checkpoint_dir = checkpoint_dir
        self.max_hours = timedelta(hours=max_hours)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: DictConfig) -> DictConfig:
        """Adds specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg.auto_resume = omegaconf_select(cfg, "auto_resume", default={})
        cfg.auto_resume.enabled = omegaconf_select(cfg, "auto_resume.enabled", default=False)
        cfg.auto_resume.max_hours = omegaconf_select(cfg, "auto_resume.max_hours", default=36)

        return cfg

    def find_checkpoint(self, cfg: DictConfig):
        """Finds a valid checkpoint that matches the arguments

        Args:
            cfg (DictConfig): DictConfig containing all settings of the model.
        """

        current_time = datetime.now()

        candidates = []
        for rootdir, _, files in os.walk(self.checkpoint_dir):
            rootdir = Path(rootdir)
            if files:
                # skip checkpoints that are empty
                try:
                    # Check for all checkpoints in the folder
                    checkpoint_files = [rootdir / f for f in files if f.endswith(".ckpt")]
                    if not checkpoint_files:
                        continue
                    # Sort by modification time to get the latest one
                    checkpoint_files.sort(key=os.path.getmtime)
                    checkpoint_file = checkpoint_files[-1]
                except:
                    continue

                creation_time = datetime.fromtimestamp(os.path.getctime(checkpoint_file))
                if current_time - creation_time < self.max_hours:
                    ck = Checkpoint(
                        creation_time=creation_time,
                        args=rootdir / "args.json",
                        checkpoint=checkpoint_file,
                    )
                    candidates.append(ck)

        if candidates:
            # sort by most recent
            candidates = sorted(candidates, key=lambda ck: ck.creation_time, reverse=True)

            for candidate in candidates:
                candidate_cfg = DictConfig(json.load(open(candidate.args)))
                if all(
                    omegaconf_select(candidate_cfg, param, None)
                    == omegaconf_select(cfg, param, None)
                    for param in AutoResumer.SHOULD_MATCH
                ):
                    wandb_run_id = getattr(candidate_cfg, "wandb_run_id", None)
                    return candidate.checkpoint, wandb_run_id

        return None, None
