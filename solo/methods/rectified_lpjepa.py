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
from solo.losses.rectified_lpjepa import (
    rectified_lp_jepa_loss,
    determine_sigma_for_lp_dist,
    choose_sigma_for_unit_var,
)
from solo.utils.metrics import l1_sparsity_metric, l0_sparsity_metric
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select

import math

# =========================
# Projection Vector Generation
# =========================
class Projections:
    @staticmethod
    def generate_random_projections(num_projections, D, device=None, dtype=None):
        """
        Generates a set of random, normalized projection vectors on the unit sphere.
        """
        P_directions = torch.randn(num_projections, D, device=device, dtype=dtype)
        P_directions = P_directions / torch.norm(P_directions, dim=1, keepdim=True)
        return P_directions

    @staticmethod
    def generate_svd_projections(z1, z2):
        """
        Computes the right-singular vectors (V^T) of the centered feature matrices z1 and z2.
        Attempts torch.linalg.svd first, falling back to LOBPCG if it fails.
        """
        # Force float32 for SVD/Eigen computations to avoid precision issues
        with torch.amp.autocast('cuda', enabled=False):
            z1 = z1.detach().float()
            z2 = z2.detach().float()

            # Center each batch (remove mean)
            z1_centered = z1 - z1.mean(dim=0)
            z2_centered = z2 - z2.mean(dim=0)

            try:
                # 1. Attempt standard SVD: z = U S V^T (we only need V^T)
                _, _, Vt_z1 = torch.linalg.svd(z1_centered, full_matrices=False)
                _, _, Vt_z2 = torch.linalg.svd(z2_centered, full_matrices=False)
                return Vt_z1, Vt_z2

            except Exception:
                # 2. Fallback to LOBPCG if SVD fails (e.g., convergence issues)
                B, D = z1_centered.shape
                k = min(B, D)

                
                A1 = torch.matmul(z1_centered.T, z1_centered)
                A2 = torch.matmul(z2_centered.T, z2_centered)

                # Initial guess for eigenvectors
                X1 = torch.randn(D, k, device=z1.device, dtype=torch.float32)
                X2 = torch.randn(D, k, device=z1.device, dtype=torch.float32)

                try:
                    # Find top-k eigenvectors
                    _, V1 = torch.lobpcg(A1, X=X1, largest=True)
                    _, V2 = torch.lobpcg(A2, X=X2, largest=True)
                    Vt_z1, Vt_z2 = V1.T, V2.T
                except Exception:
                    # 3. Final fallback to standard eigh
                    _, V1 = torch.linalg.eigh(A1)
                    _, V2 = torch.linalg.eigh(A2)
                    Vt_z1 = V1.T.flip(0)[:k]
                    Vt_z2 = V2.T.flip(0)[:k]

                return Vt_z1, Vt_z2
    

    @staticmethod
    def get_projection_vectors(
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_projections: int,
        projection_vectors_type: str,
        proj_output_dim: int,
    ):
        """
        Main entry point to get projection vectors based on the specified type.
        Supports: 'random', 'torch_svd_and_random', 'torch_svd_bottom_half_eigen_and_random'.
        """
        if projection_vectors_type == 'torch_svd_and_random':
            # Combine top eigenvectors with random projections
            Vt_z1, Vt_z2 = Projections.generate_svd_projections(z1, z2)
            random_projs = Projections.generate_random_projections(
                num_projections - Vt_z1.size(0), proj_output_dim, device=z1.device, dtype=z1.dtype
            )
            return [torch.vstack([Vt_z1, random_projs]), torch.vstack([Vt_z2, random_projs])]

        elif projection_vectors_type == 'torch_svd_bottom_half_eigen_and_random':
            # Combine bottom-half eigenvectors with random projections
            Vt_z1, Vt_z2 = Projections.generate_svd_projections(z1, z2)
            Vt_z1_bh = Vt_z1[Vt_z1.size(0)//2:]
            Vt_z2_bh = Vt_z2[Vt_z2.size(0)//2:]
            random_projs = Projections.generate_random_projections(
                num_projections - Vt_z1_bh.size(0), proj_output_dim, device=z1.device, dtype=z1.dtype
            )
            return [torch.vstack([Vt_z1_bh, random_projs]), torch.vstack([Vt_z2_bh, random_projs])]

        elif projection_vectors_type == 'random':
            # Purely random projections
            return Projections.generate_random_projections(
                num_projections, proj_output_dim, device=z1.device, dtype=z1.dtype
            )
        else:
            raise ValueError(f"Unsupported projection_vectors_type: {projection_vectors_type}")

# =========================
# Rectified LpJEPA Method
# =========================
class RectifiedLpJEPA(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Implements Rectified LpJEPA: Joint-Embedding Predictive Architectures
        with Sparse and Maximum-Entropy Representations.
        """
        super().__init__(cfg)

        # Loss weights
        self.invariance_loss_weight: float = cfg.method_kwargs.invariance_loss_weight
        self.rdm_reg_loss_weight: float = cfg.method_kwargs.rdm_reg_loss_weight

        # Distribution and Projection parameters
        self.target_distribution: str = cfg.method_kwargs.target_distribution
        self.num_projections = cfg.method_kwargs.num_projections
        self.projection_vectors_type: str = cfg.method_kwargs.projection_vectors_type
        self.mean_shift_value: float = cfg.method_kwargs.mean_shift_value
        self.lp_norm_parameter: float = cfg.method_kwargs.lp_norm_parameter
        
        # Determine target scale (chosen_sigma)
        self.mode_of_sigma: str = cfg.method_kwargs.mode_of_sigma
        if self.mode_of_sigma == "sigma_GN":
            # Scale GN_p to unit variance before rectification
            self.chosen_sigma = determine_sigma_for_lp_dist(self.lp_norm_parameter)
        elif self.mode_of_sigma == "sigma_RGN":
            # Scale GN_p such that ReLU(GN_p) has unit variance
            self.chosen_sigma = choose_sigma_for_unit_var(self.lp_norm_parameter, self.mean_shift_value)
        else:
            raise ValueError(f"Invalid mode of sigma: {self.mode_of_sigma}")
        
        print(f"Chosen sigma for {self.target_distribution} with mean shift {self.mean_shift_value} and p_norm {self.lp_norm_parameter} is {self.chosen_sigma}")

        # Projector configuration
        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.proj_output_dim = proj_output_dim
        self.projector_type: str = cfg.method_kwargs.projector_type

        # Define Projector Architecture
        if self.projector_type == "mlp":
            # Standard 3-layer MLP projector
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        elif self.projector_type == "rectified_mlp":
            # 3-layer MLP projector with final ReLU for non-negativity
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
        else:
            raise ValueError(f"Invalid projector type: {self.projector_type}")

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """
        Adds method-specific default values and checks to the config.
        """
        cfg = super(RectifiedLpJEPA, RectifiedLpJEPA).add_and_assert_specific_cfg(cfg)

        # Default loss weights
        cfg.method_kwargs.invariance_loss_weight = omegaconf_select(cfg, "method_kwargs.invariance_loss_weight", 25.0)
        cfg.method_kwargs.rdm_reg_loss_weight = omegaconf_select(cfg, "method_kwargs.rdm_reg_loss_weight", 125.0)
        
        # Default distribution/projection settings
        cfg.method_kwargs.num_projections = omegaconf_select(cfg, "method_kwargs.num_projections", 8192)
        cfg.method_kwargs.projection_vectors_type = omegaconf_select(cfg, "method_kwargs.projection_vectors_type", "random")
        cfg.method_kwargs.mean_shift_value = omegaconf_select(cfg, "method_kwargs.mean_shift_value", 0.0)
        cfg.method_kwargs.lp_norm_parameter = omegaconf_select(cfg, "method_kwargs.lp_norm_parameter", 1.0)
        cfg.method_kwargs.mode_of_sigma = omegaconf_select(cfg, "method_kwargs.mode_of_sigma", "sigma_GN")
        cfg.method_kwargs.projector_type = omegaconf_select(cfg, "method_kwargs.projector_type", "rectified_mlp")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """
        Returns the list of learnable parameters for the optimizer.
        """
        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Performs the forward pass: Backbone -> Projector.
        """
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        
        # Optional projector classifier for online evaluation
        if self.projector_classifier is not None:
            projector_logits = self.projector_classifier(z.detach())
            out.update({"projector_logits": projector_logits})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Main training step: computes invariance and RDMReg losses.
        """
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        do_log = self.global_step % self.logging_interval == 0

        # Generate projection vectors for RDMReg
        projection_vectors = Projections.get_projection_vectors(
            z1, z2, self.num_projections, self.projection_vectors_type, self.proj_output_dim
        )

        # Compute Rectified LpJEPA Loss
        loss_val, sim_l, reg_l = rectified_lp_jepa_loss(
            z1, z2, projection_vectors,
            target_distribution=self.target_distribution,
            invariance_loss_weight=self.invariance_loss_weight,
            rdm_reg_loss_weight=self.rdm_reg_loss_weight,
            mean_shift_value=self.mean_shift_value,
            lp_norm_parameter=self.lp_norm_parameter,
            chosen_sigma=self.chosen_sigma,
        )

        # Logging
        self.log("train_rectified_lp_jepa_loss", loss_val, on_epoch=True, sync_dist=True)
        self.log("train_invariance_loss", sim_l, on_epoch=True, sync_dist=True)
        self.log("train_rdm_reg_loss", reg_l, on_epoch=True, sync_dist=True)

        if do_log:
            # Gather across GPUs for global statistics logging
            from solo.utils.misc import gather
            from solo.utils.metrics import variance_loss, covariance_loss
            z1_gathered, z2_gathered = gather(z1), gather(z2)
            # Log VICReg terms (Variance and Covariance)
            self.log("train_variance_loss", variance_loss(z1_gathered, z2_gathered), on_epoch=True, sync_dist=True)
            self.log("train_covariance_loss", covariance_loss(z1_gathered, z2_gathered), on_epoch=True, sync_dist=True)
            # Log sparsity metrics
            self.log("train_l1_sparsity_metric", l1_sparsity_metric(z1), on_epoch=True, sync_dist=True)
            self.log("train_l0_sparsity_metric", l0_sparsity_metric(z1), on_epoch=True, sync_dist=True)

        # Optional online classification evaluation
        projector_class_loss = torch.tensor(0.0, device=self.device)
        if self.projector_classifier is not None:
            _, _, targets = batch
            proj_metrics1 = self._projector_classifier_step(z1, targets)
            proj_metrics2 = self._projector_classifier_step(z2, targets)
            if proj_metrics1 and proj_metrics2:
                projector_class_loss = (proj_metrics1["proj_loss"] + proj_metrics2["proj_loss"]) / 2
                if do_log:
                    self.log("train_proj_loss", projector_class_loss, on_epoch=True, sync_dist=True)
                    self.log("train_proj_acc1", (proj_metrics1["proj_acc1"] + proj_metrics2["proj_acc1"]) / 2, on_epoch=True, sync_dist=True)

        return loss_val + class_loss + projector_class_loss
