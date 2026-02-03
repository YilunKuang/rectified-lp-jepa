# Rectified LpJEPA: Minimal Example
*Date: 2026-02-03*
---

This document provides a minimal, self-contained implementation of **Rectified LpJEPA**. It induces sparse and non-negative representations by regularizing feature distributions towards a **Rectified Generalized Gaussian (RGG)** distribution.

This guide is structured like a Jupyter Notebook. If you copy each code block into a cell and run them sequentially, you will end up pretraining a ResNet-18 on CIFAR-100 for 100 epochs with sparse representations. This file also goes through the key sections of our method as well!

Before starting, install the minimal dependencies:
```bash
pip install torch torchvision timm tqdm
```

---

## 1. Imports and Utilities
First, we import our core utilities. We use `timm` for the backbone and standard `torchvision` for data handling. They are awesome!

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math, tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from torch.amp import GradScaler, autocast
```

---

## 2. The Rectified LpJEPA Model
We define a 3-layer MLP projector that maintains a constant width (512 for ResNet-18, since the output dimension of ResNet-18 is 512). Crucially, the projector ends with a **ReLU**, ensuring the learned representations are non-negative. Refer to Section 4 of our paper for more details. Adding a ReLU here is actually quite important: this ReLU is the main mechanism that produces \(\ell_0\) sparsity.


```python
class Projector(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU() # Final ReLU for non-negativity
        )
    def forward(self, x):
        return self.model(x)

class RectifiedLpJEPA(nn.Module):
    def __init__(self, backbone_name="resnet18", dim=512):
        super().__init__()
        # Load backbone and remove the final classification layer
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.projector = Projector(dim=dim)

    def forward(self, x):
        feats = self.backbone(x)
        z = self.projector(feats)
        return z
```

---

## 3. Sampling the Sparse Prior (RGG)
To induce sparsity, we align our features to a **Rectified Generalized Gaussian**. Decreasing the Mean Shift ($\mu$) pushes more of the distribution below zero, resulting in higher sparsity after rectification.

```python
def sample_rgg(shape, p=1.0, mu=0.0, device='cpu'):
    """
    Samples from Rectified Generalized Gaussian: ReLU(mu + sigma * GN_p).
    p=1.0 is Rectified Product Laplace (Sparsity prior).
    """
    # sigma for unit variance of GN_p before rectification
    sigma = math.sqrt(math.gamma(1/p) / math.gamma(3/p)) / (p**(1/p))
    
    # Sample Generalized Gaussian GN_p(0, 1)
    sign = torch.empty(shape, device=device).bernoulli_(0.5) * 2 - 1
    gamma_dist = torch.distributions.Gamma(concentration=1.0/p, rate=1.0)
    g = gamma_dist.sample(shape).to(device)
    gn_samples = sign * (p * g).pow(1.0/p)
    
    return torch.relu(mu + sigma * gn_samples)
```

---

## 4. The RDMReg Loss
We use **Sliced Wasserstein Distance (SWD)** to match the feature distribution to our RGG samples. This is based on the **Cramér–Wold theorem**. By sorting random projections, we can efficiently compute the distance between two univariate distributions.


```python
def rdmreg_loss(z, target_samples, num_projections=256):
    B, D = z.shape
    # 1. Generate random projections (normalized)
    projections = torch.randn(num_projections, D, device=z.device)
    projections = projections / projections.norm(dim=1, keepdim=True)
    
    # 2. Project features and target samples
    proj_z = torch.matmul(z, projections.T)
    proj_target = torch.matmul(target_samples, projections.T)
    
    # 3. Sort and compute Wasserstein-2 distance
    proj_z_sorted, _ = torch.sort(proj_z, dim=0)
    proj_target_sorted, _ = torch.sort(proj_target, dim=0)
    
    return torch.mean((proj_z_sorted - proj_target_sorted)**2)
```

---

## 5. Data Loading (CIFAR-100)
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

transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_ds = TwoViewDataset(datasets.CIFAR100(root='./data', train=True, download=True), transform)
# Note: num_workers=0 is used to avoid multiprocessing errors in Jupyter Notebooks
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
```

---

## 6. Main Training Loop

Our main loss is a combination of the **Invariance Loss** and the **RDMReg Loss**. The invariance loss, used in self-supervised learning, ensures that similar features stay close together, while the RDMReg loss adds a sparsity prior and also acts as a regularizer that helps prevent collapse. Finally, we assemble everything and optimize the combined objective.


```python
# Hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
use_cuda = torch.cuda.is_available()
p, mu = 1.0, 0.0  # Laplace prior with 0 mean shift
lamb_inv, lamb_reg = 25.0, 125.0 # Weights from Table 1

model = RectifiedLpJEPA().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = GradScaler(enabled=use_cuda)

for epoch in range(100):
    model.train()
    # Use leave=False to keep the notebook output clean
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    epoch_loss = 0
    for x1, x2, _ in pbar:
        x1, x2 = x1.to(device), x2.to(device)
        
        with autocast(device_type=('cuda' if use_cuda else 'cpu'), enabled=use_cuda):
            z1, z2 = model(x1), model(x2)
            
            # Generate target samples and compute losses
            target = sample_rgg(z1.shape, p=p, mu=mu, device=device)
            inv_loss = F.mse_loss(z1, z2)
            reg_loss = (rdmreg_loss(z1, target) + rdmreg_loss(z2, target)) / 2
            loss = (lamb_inv * inv_loss) + (lamb_reg * reg_loss)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Log sparsity (fraction of zero activations)
        sparsity = (z1 == 0).float().mean().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "sparse": f"{sparsity:.2%}"})
        epoch_loss += loss.item()
    
    # Print a clean summary for the epoch
    print(f"Epoch {epoch} | Avg Loss: {epoch_loss/len(train_loader):.4f} | Final Sparsity: {sparsity:.2%}")
```

---

## Key Parameters & Sparsity Control
- **Mean Shift ($\mu$)**: Your **controllable sparsity parameter**. 
    - **Lower $\mu$ (e.g., -2.0)**: Induces **higher sparsity** (fewer active features).
    - **Higher $\mu$ (e.g., 0.5)**: Induces **lower sparsity** (denser representations).
- **Shape Parameter ($p$)**: 
    - **$p=1.0$**: Rectified Product Laplace (sparse prior).
    - **$p=2.0$**: Rectified Gaussian.
- **Weights**: We recommend `lamb_inv=25.0` and `lamb_reg=125.0` for optimal performance.
