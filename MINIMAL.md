# Rectified LpJEPA: Minimal Example

This document presents a minimal, self-contained implementation of **Rectified LpJEPA**, which learns sparse, non-negative representations by regularizing feature distributions towards a **Rectified Generalized Gaussian (RGG)** distribution.

This guide is structured like a Jupyter Notebook. By copying each code block into a cell and running them sequentially, you can pretrain a Rectified LpJEPA model with a ResNet-18 backbone on CIFAR-100 for 100 epochs. The guide also walks through the key components of our method.

Please note that this document is not intended to provide an optimal or fully tuned implementation of Rectified LpJEPA on CIFAR-100. Achieving leaderboard-level performance in self-supervised learning typically requires many additional details and careful tuning. Instead, we present a minimal and simplified implementation whose goal is to clearly illustrate how the method works.

For a more complete and performant implementation, please use the following training script:

```python
python3 main_pretrain.py \
  --config-path scripts/pretrain/cifar/ \
  --config-name=rectified_lpjepa_cifar100.yaml \
  ++wandb.entity=<ENTITY> \
  ++wandb.project=<PROJECT> \
  ++wandb.enabled=true
```

Alright, with that, let's get started!

---
## 1. Imports and Utilities

Before starting, install the minimal dependencies:
```bash
pip install torch torchvision tqdm wandb
```

Now we start by importing our core utilities. We use the `torchvision` library for the backbone and data handling.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
```

We also provide a minimal [LARS](https://arxiv.org/abs/1708.03888) optimizer implementation for training. The full implementation can be found in `solo/utils/lars.py`.

```python
class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-4, eta=0.02, eps=1e-8, exclude_bias_n_norm=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps, exclude_bias_n_norm=exclude_bias_n_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            eps = group["eps"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None: continue
                d_p = p.grad

                if p.ndim != 1 or not group["exclude_bias_n_norm"]:
                    p_norm = torch.norm(p)
                    g_norm = torch.norm(d_p)
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = eta * p_norm / (g_norm + p_norm * weight_decay + eps)
                        d_p = d_p.add(p, alpha=weight_decay).mul(lars_lr)

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p)
                p.add_(buf, alpha=-lr)
```
To evaluate both sparsity and performance, we use the following accuracy, l1, and l0 sparsity metrics.

```python
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def l1_sparsity_metric(val_feats: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Calculates the mean l1 sparsity metric: (1/D) * (||z_row||_1 / ||z_row||_2)^2
    """
    with torch.no_grad():
        D = val_feats.shape[1]
        l1_norms = torch.linalg.norm(val_feats, ord=1, dim=1)
        l2_norms = torch.linalg.norm(val_feats, ord=2, dim=1)
        l1_sparsity_per_sample = (1.0 / D) * (l1_norms / (l2_norms + eps))**2
        return l1_sparsity_per_sample.mean().item()

def l0_sparsity_metric(val_feats: torch.Tensor) -> float:
    """
    Calculates the mean l0 sparsity metric: fraction of nonzero elements per sample
    """
    with torch.no_grad():
        D = val_feats.shape[1]
        l0_sparsity_per_sample = (val_feats != 0).sum(dim=1).float() / D
        return l0_sparsity_per_sample.mean().item()
```


---

## 2. Rectified LpJEPA


### Model Architecture (`__init__` and `forward`)

Following common practice in self-supervised learning, we use a 3-layer MLP projector with width 512 on top of the ResNet-18 encoder backbone. In our design, the MLP projector has to end with a **ReLU**, which is essential for the correctness of our method. Refer to Section 4 of our paper for more details.

It's also possible to use ViT as the encoder backbone. See `scripts/pretrain/cifar/rectified_lpjepa_cifar100_vit.yaml` for the configurations.

### Rectified Generalized Gaussian (RGG) Distribution (`sample_product_laplace` and `sample_rgg`)

To induce sparsity, we align our features to a **Rectified Generalized Gaussian**. As we mentioned in the paper, we usually fix $\sigma=\Gamma(1/p)^{1/2}/(p^{1/p}\cdot\Gamma(3/p)^{1/2})$. In general, decreasing $p$ and $\mu$ results in higher sparsity both in terms of the L0 and L1 norm metrics defined in our paper. 

To avoid the extra costs of tuning, we recommend using $p=1$ by default, which corresponds to the **Rectified Laplace** distribution that enjoys both L0 sparsity guarantees and L1 norm constraints. 

### Rectified Distribution Matching Regularization (RDMReg) (`rdmreg_loss`)

We instantiate RDMReg using the two-sample **Sliced 2-Wasserstein Distance (SWD)** to align the empirical feature distribution with samples from our RGG target. SWD approximates a high-dimensional distributional discrepancy by averaging 1D Wasserstein distances over many random projection directions. This is motivated by the **Cramér–Wold theorem**, which states that a distribution is characterized by its collection of one-dimensional projections—so matching many projected marginals encourages the full distributions to align.

### SSL Loss (`ssl_loss`)

Our main loss is a combination of the **Invariance Loss** and the **RDMReg Loss**. The invariance loss, used in self-supervised learning, ensures that similar features stay close together, while the RDMReg loss preserves the maximum-entropy guarantees under the expected Lp norm constraints to prevent collapse while also leads to explicit L0 sparsity by the choice of the target RGG distribuion.

```python
class RectifiedLpJEPA(nn.Module):
    def __init__(self, proj_hidden_dim=512, proj_output_dim=512):
        super().__init__()

        # This is the standard ResNet18 architecture while removing the final fc layer and maxpool layer.
        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(512, proj_hidden_dim), # the output dim of the ResNet18 backbone is 512
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.ReLU() # Final ReLU for non-negativity and l0 sparsity
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.projector(feat)
        return feat, z

    def sample_product_laplace(self, shape, device, loc=0.0, scale=1/math.sqrt(2)):
        """
        Sample from the product Laplace distribution directly on the target device
        using torch.distributions.Laplace.
        """
        # shape is (Batch Size, Dimension)
        loc_t = torch.tensor(loc, device=device)
        scale_t = torch.tensor(scale, device=device)
        laplace_dist = torch.distributions.Laplace(loc=loc_t, scale=scale_t)
        return laplace_dist.sample(shape)
    
    # Sample Rectified Generalized Gaussian
    def sample_rgg(self, shape, p=1.0, mu=0.0, device='cpu'):
        """
        Samples from Rectified Generalized Gaussian: ReLU(mu + sigma * GN_p).
        p=1.0 is Rectified Product Laplace (Sparsity prior).
        """
        assert p > 0.0, "p must be > 0.0"
        
        # sigma_GN for unit variance of GN_p before rectification. 
        # See the choose_sigma_for_unit_var() in solo/losses/rectified_lpjepa.py for the alternative choice of sigma_RGN
        sigma = math.sqrt(math.gamma(1/p) / math.gamma(3/p)) / (p**(1/p))

        if p == 1.0:
            return torch.relu(self.sample_product_laplace(shape, device, loc=mu, scale=sigma))
        elif p == 2.0:
            return torch.relu(mu + sigma * torch.randn(shape, device=device))
        else:
            # Sample Generalized Gaussian GN_p(0, 1). This is in general slower than Rectified Laplace and Rectified Gaussian.
            sign = torch.empty(shape, device=device).bernoulli_(0.5) * 2 - 1
            gamma_dist = torch.distributions.Gamma(concentration=1.0/p, rate=1.0)
            g = gamma_dist.sample(shape).to(device)
            gn_samples = sign * (p * g).pow(1.0/p)
            
            return torch.relu(mu + sigma * gn_samples)

    def rdmreg_loss(self, z, p=1.0, mu=0.0, num_projections=8192):
        B, D = z.shape
        device = z.device
        target_samples = self.sample_rgg((B, D), p, mu, device)
        
        # 1. Generate random projections (normalized)
        projections = torch.randn(num_projections, D, device=device)
        projections = projections / projections.norm(dim=1, keepdim=True)
        
        # 2. Project features and target samples
        proj_z = torch.matmul(z, projections.T)
        proj_target = torch.matmul(target_samples, projections.T)
        
        # 3. Sort and compute Wasserstein-2 distance
        proj_z_sorted, _ = torch.sort(proj_z, dim=0)
        proj_target_sorted, _ = torch.sort(proj_target, dim=0)

        return torch.mean((proj_z_sorted - proj_target_sorted)**2)

    def ssl_loss(self, x1, x2, p=1.0, mu=0.0, num_projections=8192, lamb_inv=25.0, lamb_reg=250.0):
        f1, z1 = self.forward(x1)
        f2, z2 = self.forward(x2)
        
        # Invariance Loss
        inv_loss = F.mse_loss(z1, z2)
        # RDMReg Loss
        reg_loss = (self.rdmreg_loss(z1, p, mu, num_projections) + self.rdmreg_loss(z2, p, mu, num_projections)) / 2
        # SSL Loss = \lambda_1 * Invariance Loss + \lambda_2 * RDMReg Loss
        ssl_loss = lamb_inv * inv_loss + lamb_reg * reg_loss
        return ssl_loss, f1, f2, z1, z2, inv_loss, reg_loss
```

---

## 3. Data Loading (CIFAR-100)

We use a simple two-view augmentation strategy: each image is transformed twice to form a positive pair \((x_1, x_2)\). This setup is essential for self-supervised learning. The two images act as positives because they are different augmented views of the same underlying visual content, created by applying random transformations to the same image.

```python
class TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    def __getitem__(self, i):
        img, label = self.base_dataset[i]
        return self.transform(img), self.transform(img), label
    def __len__(self):
        return len(self.base_dataset)

def get_dataloader():
    # solo-learn uses BICUBIC and specific CIFAR100 normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.RandomSolarize(128)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    val_transform = transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    train_base = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    train_dataset = TwoViewDataset(train_base, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
```

## 4. Configurations

Here are the hyperparameters we will use for this guide.

```python
WANDB_ENTITY = "INSERT_YOUR_WANDB_ENTITY" # wandb for logging
WANDB_PROJECT = "INSERT_YOUR_WANDB_PROJECT" # wandb for logging
BATCH_SIZE = 256 # our method works with small batch size. It's also fine to set this to 128
MAX_EPOCHS = 100 # use 1000 epochs for better performance.
LR = 0.03 # backbone + projector learning rate. To fully optimize the performance, one should use maximum update parameterization for per-layer lr scaling. We didn't do this in our project.
CLASSIFIER_LR = 0.01 # linear probe learning rate 
WEIGHT_DECAY = 1e-4 # by default, we apply weight decay following the VICReg implementation. It's also okay to set it to 0.0
LAMB_INV = 25.0 # hyper for the invariance loss
LAMB_REG = 250.0 # hyper for the RDMReg loss
P = 1.0 # p=1 corresponds to the Rectified Laplace distribution. This is the best default choice.
MU = 0.0 # decrease mu to learn sparser representations. In our experiments, we sweep mu in the closed interval [-3, 1].
NUM_PROJECTIONS = 8192 # number of random projection vectors. In practice, 8192 is more than enough. In fact, setting this to 512 also works
PROJ_HIDDEN_DIM = 512 # projector dimension. Increase this for better performance.
PROJ_OUTPUT_DIM = 512 # projector dimension. Increase this for better performance.
```

---

## 5. Main Training Loop

With all the essential components defined, we assemble everything and perform training.


```python

device = torch.device("cuda")
wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name="minimal_rectified_lpjepa_cifar100")

model = RectifiedLpJEPA(PROJ_HIDDEN_DIM, PROJ_OUTPUT_DIM).to(device)
classifier = nn.Linear(512, 100).to(device)
proj_classifier = nn.Linear(PROJ_OUTPUT_DIM, 100).to(device)

optimizer = LARS([
    {'params': model.parameters(), 'lr': LR},
    {'params': classifier.parameters(), 'lr': CLASSIFIER_LR, 'weight_decay': 0},
    {'params': proj_classifier.parameters(), 'lr': CLASSIFIER_LR, 'weight_decay': 0}
], lr=LR, weight_decay=WEIGHT_DECAY)

train_loader, val_loader = get_dataloader()
total_steps = MAX_EPOCHS * len(train_loader)
warmup_steps = 10 * len(train_loader)

for epoch in range(MAX_EPOCHS):
    model.train(); classifier.train(); proj_classifier.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, (x1, x2, y) in enumerate(pbar):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        # LR Schedule
        curr_step = epoch * len(train_loader) + step
        if curr_step < warmup_steps:
            curr_lr = LR * curr_step / warmup_steps
        else:
            curr_lr = LR * 0.5 * (1 + math.cos(math.pi * (curr_step - warmup_steps) / (total_steps - warmup_steps)))
        
        optimizer.param_groups[0]['lr'] = curr_lr
        optimizer.param_groups[1]['lr'] = CLASSIFIER_LR
        optimizer.param_groups[2]['lr'] = CLASSIFIER_LR
        
        # Forward & Loss
        ssl_loss, f1, _, z1, _, inv_loss, reg_loss = model.ssl_loss(x1, x2, P, MU, NUM_PROJECTIONS, LAMB_INV, LAMB_REG)
        
        logits = classifier(f1.detach())
        p_logits = proj_classifier(z1.detach())
        
        class_loss = F.cross_entropy(logits, y)
        proj_class_loss = F.cross_entropy(p_logits, y)
        
        loss = ssl_loss + class_loss + proj_class_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            acc1, = accuracy(logits, y)
            proj_acc1, = accuracy(p_logits, y)
            l1_sparse = l1_sparsity_metric(z1)
            l0_sparse = l0_sparsity_metric(z1)
            wandb.log({"loss": loss.item(), "ssl_loss": ssl_loss.item(), "class_loss": class_loss.item(), 
                        "proj_class_loss": proj_class_loss.item(),
                        "inv_loss": inv_loss.item(), "reg_loss": reg_loss.item(), 
                        "acc1": acc1.item(), "proj_acc1": proj_acc1.item(),
                        "l1_sparsity": l1_sparse, "l0_sparsity": l0_sparse,
                        "lr": curr_lr, "epoch": epoch})
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc1": f"{acc1.item():.1f}", "p_acc1": f"{proj_acc1.item():.1f}"})

    # Validation
    model.eval(); classifier.eval(); proj_classifier.eval()
    val_acc1, val_proj_acc1 = 0, 0
    val_l1, val_l0 = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            feat, z = model(x)
            logits = classifier(feat)
            p_logits = proj_classifier(z)
            a1, = accuracy(logits, y)
            pa1, = accuracy(p_logits, y)
            val_acc1 += a1.item()
            val_proj_acc1 += pa1.item()
            val_l1 += l1_sparsity_metric(z)
            val_l0 += l0_sparsity_metric(z)
    
    val_acc1 /= len(val_loader)
    val_proj_acc1 /= len(val_loader)
    val_l1 /= len(val_loader)
    val_l0 /= len(val_loader)
    
    wandb.log({"val/acc1": val_acc1, "val/proj_acc1": val_proj_acc1, "val/l1_sparsity": val_l1, "val/l0_sparsity": val_l0, "epoch": epoch})
    print(f"Epoch {epoch}: Val Acc1: {val_acc1:.2f}, Val Proj Acc1: {val_proj_acc1:.2f}, Val L0: {val_l0:.3f}")

wandb.finish()
```

---

## Key Parameters & Sparsity Control
- **Mean Shift ($\mu$)**: Your **controllable sparsity parameter**. 
    - **Lower $\mu$ (e.g., -2.0)**: Induces **higher sparsity** (sparser features; more zeros in the features).
    - **Higher $\mu$ (e.g., 0.5)**: Induces **lower sparsity** (denser features; less zeros in the features).
- **Shape Parameter ($p$)**: 
    - **$p=1.0$**: Rectified Laplace (default choice).
    - **$p=2.0$**: Rectified Gaussian.
- **Weights**: We recommend `lamb_inv=25.0` and `lamb_reg=125.0` for optimal performance.
