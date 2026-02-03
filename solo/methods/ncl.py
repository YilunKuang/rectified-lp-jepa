# Copyright 2022 solo-learn development team.

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
from solo.losses.ncl import *  # same as simclr.py but copied from upstream
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select

import numpy as np



class NCL(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
                projector_type (str): type of projector to use. - MLP_w_batchorm or MLP2_wo_batchnorm
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature # it seems we use tempreture = 0.2 here

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        
        # they have weird hidden and output dimension for the projector -- let's just try 256-256; 512; 512 and 2048 3048

        self.non_neg = cfg.method_kwargs.non_neg # rep_relu etc

        # projector
        self.projector_type = cfg.method_kwargs.projector_type
        if self.projector_type == "mlp_2_wo_batchnorm":

            self.projector = nn.Sequential(  # matches reference implementation (no batchnorm, small projector)
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        elif self.projector_type == "mlp_3_w_batchnorm":
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(NCL, NCL).add_and_assert_specific_cfg(cfg) # okay we have this in other methods too! super (NCL, NCL) will return parent which would be BaseMethod. 
        cfg.method_kwargs.non_neg = omegaconf_select(cfg, "method_kwargs.non_neg", None)
        cfg.method_kwargs.projector_type = omegaconf_select(cfg, "method_kwargs.projector_type", "mlp_2_wo_batchnorm") # this is their deafult

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

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
        
        # Added to get the projector logits from the forward pass.
        # If the projector classifier exists (initialized in BaseMethod), call it.  # self.projector_classifier is from BaseMethod
        if self.projector_classifier is not None:
            # Pass z.detach() to ensure gradients for this classifier only update itself,
            # not the projector or backbone through this path.

            projector_logits = self.projector_classifier(z.detach())
        
            out.update({"projector_logits": projector_logits}) # we will now also get projector acc1 and acc5 too

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

        # Capture z before activation for non-neg ratio diagnostic
        z_pre = z.clone()

        # ------- non-negative -------
        supported_non_neg_list = [None, 'relu', 'rep_relu', 'gelu', 'sigmoid', 'softplus', 'exp', 'leakyrelu']
        assert self.non_neg in supported_non_neg_list, f"non_neg {self.non_neg} should be one of {supported_non_neg_list}"

        if self.non_neg is None:
            # leave blank to restore original contrastive learning
            pass # z=z
        if self.non_neg == 'relu': 
            z = F.relu(z)
        if self.non_neg == 'rep_relu': # alwyas use this they do in the paper too
            # reparameterized ReLU: forward is ReLU and backward is GELU
            # Similar to the gubml approach; can be reused here.
            # more friendly to backpropagation and avoids dead neurons 
            gelu_z = F.gelu(z)
            z = gelu_z - gelu_z.data + F.relu(z).data # this is all that they add

        # some other choices of activation functions that are not non-negative
        if self.non_neg == 'gelu':
            z = F.gelu(z)
        if self.non_neg == 'sigmoid':
            z = F.sigmoid(z)
        if self.non_neg == 'softplus':
            z = F.softplus(z)
        if self.non_neg == 'exp':
            z = torch.exp(z)
        if self.non_neg == 'leakyrelu':
            z = F.leaky_relu(z)

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)
        z = F.normalize(z, dim=-1)

        nce_loss = simclr_loss_func(
            z,
            indexes=indexes,
            temperature=self.temperature,
        )
        
        self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            _, X, targets = batch
            targets2 = targets.repeat(n_augs)
            
            # Ensure calculations are done on the correct device
            z_stats = z.detach()
            targets2_stats = targets2.detach()
            z_pre_stats = z_pre.detach()
            
            stats = {
                'non_neg_ratio': non_neg(z_pre_stats),
                'num_active_dim': act_dim(z_stats),
                'sparse_vals_ratio': sparsity(z_stats),
                'effective_rank': erank(z_stats),
                'orthogonality': orthogonality(z_stats),
                'semantic_consistency': semantic_consistency(z_stats, targets2_stats),
            }
            for k, v in stats.items():
                # Ensure the logged value is a tensor on the correct device for sync_dist
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, device=z.device)
                elif v.device != z.device:
                    v = v.to(z.device)
                
                self.log(k, v, on_epoch=True, on_step=False, sync_dist=True)

        # Projector Classifier Training -- copied from methods/rectified_lpjepa.py
        projector_class_loss = torch.tensor(0.0, device=self.device)
        if self.projector_classifier is not None:
             # get targets
             _, _, targets = batch
             
             proj_losses = []
             proj_acc1s = []
             proj_acc5s = []
             
             # out["z"] is a list of z tensors for each large crop
             for z_view in out["z"]:
                 metrics = self._projector_classifier_step(z_view, targets)
                 if metrics:
                     proj_losses.append(metrics["proj_loss"])
                     proj_acc1s.append(metrics["proj_acc1"])
                     proj_acc5s.append(metrics["proj_acc5"])
             
             if proj_losses:
                 projector_class_loss = sum(proj_losses) / len(proj_losses)
                 self.log("train_proj_loss", projector_class_loss, on_epoch=True, sync_dist=True)
                 
                 acc1 = sum(proj_acc1s) / len(proj_acc1s)
                 self.log("train_proj_acc1", acc1, on_epoch=True, sync_dist=True)
                 
                 acc5 = sum(proj_acc5s) / len(proj_acc5s)
                 self.log("train_proj_acc5", acc5, on_epoch=True, sync_dist=True)

        return nce_loss + class_loss + projector_class_loss
    
# ====== functions for calculating feature statistics ======

# Additional helper functions could be added here.

# ratio of non-negative values (fact check that outputs are all non-negative)
def non_neg(z): 
    return (z>=0).float().mean()

# ratio of activated dimensions along minibatch samples
def act_dim(z): # average number of active dimensions per sample
    return (z.abs() > 1e-5).float().sum(dim=1).mean()


# avereage ratio of zero-values per sample
def sparsity(z): # very similar to our implemenation of active feature threshold! They use sparsity as 1e-5
    return 1 - (z.abs()>1e-5).float().mean()
# effective rank of the feature matrix
def erank(z):
    z = z.float()
    s = torch.linalg.svdvals(z)
    s = s / s.sum()
    return -torch.sum(s * torch.log(s + 1e-6))


# semantic consistency
def semantic_consistency(features, labels, eps=1e-5, take_abs=False, topk=False):
    # find activated dimensions
    active_dim_mask = features.abs().sum(0)>0
    features  = features[:, active_dim_mask]
    features = F.normalize(features, dim=1)
    
    # Ensure features are on the same device as labels to avoid device mismatches
    if features.device != labels.device:
        features = features.to(labels.device)

    # if topk:
    #     sorted, indices = torch.sort(features.sum(dim=0), descending=True)
    #     indices = indices[sorted>1]
    #     features = features[:, indices]

    acc_per_dim = []
    for i in range(features.shape[1]): # sweep each feature dimension
        # only account for activated samples
        active_sample_mask = features.abs()[:,i] > eps
        labels_selected = labels[active_sample_mask]
        try:
            dist = labels_selected.bincount()
            dist = dist / dist.sum() # normalize to 1
            acc = dist.max().item() # ratio of the most frequent label among activatived samples
            acc_per_dim.append(acc)
        except:
            pass # sometimes it goes into err
    
    if len(acc_per_dim) == 0:
        return torch.tensor(0.0, device=features.device)
        
    mean_acc = torch.tensor(acc_per_dim, device=features.device).mean()
    return mean_acc


def orthogonality(features, eps=1e-5):
    features  = features[:,features.abs().sum(0)>0]
    n, d = features.shape
    features = F.normalize(features, dim=0)
    corr = features.T @ features
    err = (corr - torch.eye(d, device=features.device)).abs()
    err = err.mean()
    return err
        
