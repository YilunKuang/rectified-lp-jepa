# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class RepReLU(nn.Module):
    """
    Reparameterized ReLU (RepReLU):
    - Forward pass: Behaves exactly like ReLU (z = max(0, z)).
    - Backward pass: Gradients flow as if it were GELU.

    This helps avoid dead neurons during training while maintaining
    non-negative sparsity in the forward pass.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        # Calculate GELU (used for gradients)
        gelu_z = F.gelu(z)

        # Calculate ReLU (used for actual output values)
        relu_z = F.relu(z)

        return gelu_z - gelu_z.detach() + relu_z.detach()


class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
                projector_type (str): type of projector to use.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        projector_type: str = cfg.method_kwargs.projector_type
        
        if projector_type == "standard":
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        elif projector_type == "mlp3":
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        elif projector_type == "mlp3_with_one_more_relu":
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
                nn.ReLU(),
            )
        elif projector_type == "mlp3_with_one_more_rep_relu":
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
                RepReLU(),
            )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")
        cfg.method_kwargs.projector_type = omegaconf_select(
            cfg,
            "method_kwargs.projector_type",
            "standard",
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        if self.projector_classifier is not None:
            out["projector_logits"] = self.projector_classifier(z.detach())
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        if self.projector_classifier is not None:
            out["projector_logits"] = self.projector_classifier(z.detach())
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )

        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        if self.global_step % self.logging_interval == 0:
             if len(out["z"]) >= 2:
                self._log_sparsity_metrics(out["z"][0], out["z"][1], out["feats"][0], out["feats"][1])

        projector_class_loss = 0
        if self.projector_classifier is not None and "proj_loss" in out:
             projector_class_loss = sum(out["proj_loss"]) / len(out["proj_loss"])
             if self.global_step % self.logging_interval == 0:
                 self.log("train_proj_loss", projector_class_loss, on_epoch=True, sync_dist=True)
                 self.log("train_proj_acc1", sum(out["proj_acc1"])/len(out["proj_acc1"]), on_epoch=True, sync_dist=True)
                 self.log("train_proj_acc5", sum(out["proj_acc5"])/len(out["proj_acc5"]), on_epoch=True, sync_dist=True)

        return nce_loss + class_loss + projector_class_loss
