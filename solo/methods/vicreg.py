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
from solo.losses.vicreg import vicreg_loss_func
from solo.methods.base import BaseMethod
from solo.methods.ncl import act_dim, erank, non_neg, orthogonality, semantic_consistency, sparsity
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


class VICReg(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                sim_loss_weight (float): weight of the invariance term.
                var_loss_weight (float): weight of the variance term.
                cov_loss_weight (float): weight of the covariance term.
                projector_type (str): type of projector to use.
        """

        super().__init__(cfg)

        self.sim_loss_weight: float = cfg.method_kwargs.sim_loss_weight
        self.var_loss_weight: float = cfg.method_kwargs.var_loss_weight
        self.cov_loss_weight: float = cfg.method_kwargs.cov_loss_weight

        self.non_neg: bool = cfg.method_kwargs.non_neg
        self.projector_type: str = cfg.method_kwargs.projector_type

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        projector_type: str = cfg.method_kwargs.projector_type

        if projector_type == "mlp3":
            # projector
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

        cfg = super(VICReg, VICReg).add_and_assert_specific_cfg(cfg)

        cfg.method_kwargs.non_neg = omegaconf_select(cfg, "method_kwargs.non_neg", False) # this is for non-negative

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")

        cfg.method_kwargs.sim_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.sim_loss_weight",
            25.0,
        )
        cfg.method_kwargs.var_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.var_loss_weight",
            25.0,
        )
        cfg.method_kwargs.cov_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.cov_loss_weight",
            1.0,
        )
        cfg.method_kwargs.projector_type = omegaconf_select(
            cfg,
            "method_kwargs.projector_type",
            "mlp3",
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

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        if self.projector_classifier is not None:
            out["projector_logits"] = self.projector_classifier(z.detach())
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- non-negative -------
        if self.non_neg:
            z1 = F.relu(z1)
            z2 = F.relu(z2)

        # ------- vicreg loss -------
        vicreg_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        self.log("train_vicreg_loss", vicreg_loss, on_epoch=True, sync_dist=True)

        if self.global_step % self.logging_interval == 0:
            self._log_sparsity_metrics(z1, z2, out["feats"][0], out["feats"][1])

        projector_class_loss = 0
        if self.projector_classifier is not None and "proj_loss" in out:
             projector_class_loss = sum(out["proj_loss"]) / len(out["proj_loss"])
             if self.global_step % self.logging_interval == 0:
                 self.log("train_proj_loss", projector_class_loss, on_epoch=True, sync_dist=True)
                 self.log("train_proj_acc1", sum(out["proj_acc1"])/len(out["proj_acc1"]), on_epoch=True, sync_dist=True)
                 self.log("train_proj_acc5", sum(out["proj_acc5"])/len(out["proj_acc5"]), on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            _, X, targets = batch
            n_augs = self.num_large_crops + self.num_small_crops
            targets2 = targets.repeat(n_augs)

            # For stats, apply non-neg if requested
            z_all = []
            for z_view in out["z"]:
                if self.non_neg:
                    z_view = F.relu(z_view)
                z_all.append(z_view)

            z_stats = torch.cat(z_all).detach()
            targets2_stats = targets2.detach()

            stats = {
                "non_neg_ratio": non_neg(z_stats),
                "num_active_dim": act_dim(z_stats),
                "sparse_vals_ratio": sparsity(z_stats),
                "effective_rank": erank(z_stats),
                "orthogonality": orthogonality(z_stats),
                "semantic_consistency": semantic_consistency(z_stats, targets2_stats),
            }
            for k, v in stats.items():
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=self.device)
                elif v.device != self.device:
                    v = v.to(self.device)

                self.log(k, v, on_epoch=True, on_step=False, sync_dist=True)

        return vicreg_loss + class_loss + projector_class_loss
