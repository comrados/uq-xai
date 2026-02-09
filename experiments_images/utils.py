from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from skimage.metrics import structural_similarity as ssim
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from torchvision import transforms

# BASE PATHS
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "plantvillage_raw"
PROCESSED_DIR: Path = DATA_DIR / "plantvillage"
TRAIN_DIR: Path = PROCESSED_DIR / "train"
VAL_DIR: Path = PROCESSED_DIR / "val"
TEST_DIR: Path = PROCESSED_DIR / "test"
MODEL_DIR: Path = BASE_DIR / "models"
MODEL_PATH: Path = MODEL_DIR / "plant_cnn.pt"
RESULTS_DIR: Path = BASE_DIR / "results"
EPIS_PATH: Path = RESULTS_DIR / "epistemic_data.pkl"
SSIM_PATH: Path = RESULTS_DIR / "epistemic_ssim.pkl"
SCATTER_PLOT_PATH: Path = RESULTS_DIR / "scatter_plot.png"

# SHARED SETTINGS
IMG_SIZE: tuple[int, int] = (128, 128)
NORM_MEAN: list[float] = [0.5, 0.5, 0.5]
NORM_STD: list[float] = [0.5, 0.5, 0.5]
NUM_CLASSES: int = 3
DROPOUT: float = 0.3


class PlantCNN(nn.Module):
    """Simple CNN with VGG-style blocks and global average pooling."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        """Initialize the CNN.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability for convolution and head layers.
        """
        super().__init__()
        self.features = nn.Sequential(
            # CONV1 3 32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128-64
            nn.Dropout2d(dropout),
            
            # CONV2 32 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64-32
            nn.Dropout2d(dropout),
            
            # CONV3 64 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32-16
            nn.Dropout2d(dropout),

            # GLOBAL AVERAGE POOLING 128 1 1
            nn.AdaptiveAvgPool2d(1),  
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 128
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input batch tensor of shape (N, 3, 128, 128).
        Returns:
            Model logits of shape (N, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)


def get_device() -> torch.device:
    """Select the best available torch device.

    Returns:
        Selected torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    device: torch.device,
    model_path: Path = MODEL_PATH,
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT,
) -> PlantCNN:
    """Load a trained PlantCNN model from disk.

    Args:
        device: Torch device to load the model onto.
        model_path: Path to the model checkpoint.
        num_classes: Number of output classes.
        dropout: Dropout probability for the model.
    Returns:
        Loaded PlantCNN instance.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with image_extension/01_train_cnn.py."
        )

    model = PlantCNN(num_classes=num_classes, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def get_transform(
    img_size: tuple[int, int] = IMG_SIZE,
    mean: list[float] = NORM_MEAN,
    std: list[float] = NORM_STD,
) -> "transforms.Compose":
    """Return the default image transform pipeline.

    Args:
        img_size: Resize target (height, width).
        mean: Channel-wise normalization mean.
        std: Channel-wise normalization std.
    Returns:
        Torchvision Compose transform.
    """
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def compute_epistemic(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 50,
    device: torch.device | str = "cpu",
) -> float:
    """Compute MC Dropout epistemic uncertainty for a single image.

    Args:
        model: Model with dropout layers.
        x: (1, 3, 128, 128) single image.
        n_samples: Number of MC Dropout samples.
        device: Torch device or device string.
    Returns:
        epistemic: Epistemic variance across MC samples.
    """
    model.train()  # Enable dropout
    logits_list = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x.to(device))
            logits_list.append(logits.cpu())
    
    logits_list = torch.stack(logits_list)  # (n_samples, 1, num_classes)
    probs_list = F.softmax(logits_list, dim=-1)
    
    # Epistemic = variance across samples
    epistemic = probs_list.var(dim=0).sum().item()
    
    return epistemic

def compute_ig(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute integrated gradients attributions for a single image.

    Args:
        model: Model to explain.
        image: (1, 3, 128, 128).
        target_class: Target class index.
        device: Torch device or device string.
    Returns:
        attribution: (3, 128, 128) numpy array.
    """
    model.eval()
    ig = IntegratedGradients(model)
    
    baseline = torch.zeros_like(image)
    
    attribution = ig.attribute(
        image.to(device),
        baselines=baseline.to(device),
        target=target_class,
        n_steps=50
    )
    
    # AVERAGE OVER CHANNELS CONVERT TO NUMPY
    attr_gray = attribution.squeeze(0).mean(dim=0).cpu().numpy()
    
    return attr_gray

def compute_smoothgrad(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device | str = "cpu",
    n_samples: int = 20,
    noise_sigma: float = 0.1,
) -> np.ndarray:
    """Compute SmoothGrad attributions for a single image.

    Args:
        model: Model to explain.
        image: (1, 3, 128, 128).
        target_class: Target class index.
        device: Torch device or device string.
        n_samples: Number of noise samples to average.
        noise_sigma: Standard deviation of noise.
    Returns:
        attribution: (3, 128, 128) numpy array.
    """
    model.eval()
    grads = []

    for _ in range(n_samples):
        noisy = torch.clamp(image + torch.randn_like(image) * noise_sigma, -1, 1).to(device)
        noisy.requires_grad_(True)

        output = model(noisy)
        score = output[:, target_class].sum()

        model.zero_grad(set_to_none=True)
        if noisy.grad is not None:
            noisy.grad.zero_()
        score.backward()

        grads.append(noisy.grad.detach().cpu())

    avg_grad = torch.stack(grads).mean(dim=0)
    attr_gray = avg_grad.squeeze(0).mean(dim=0).numpy()
    return attr_gray

def compute_ssim(attr1: np.ndarray, attr2: np.ndarray) -> float:
    """Compute SSIM between two attribution maps.

    Args:
        attr1, attr2: (H, W) numpy arrays.
    Returns:
        ssim_score: SSIM value.
    """
    # Normalize to [0, 1]
    attr1 = (attr1 - attr1.min()) / (attr1.max() - attr1.min() + 1e-8)
    attr2 = (attr2 - attr2.min()) / (attr2.max() - attr2.min() + 1e-8)
    
    return ssim(attr1, attr2, data_range=1.0)

def add_gaussian_noise(image: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise to a normalized image tensor.

    Args:
        image: (C, H, W) tensor, normalized to [-1, 1].
        sigma: Noise level (0.1 = 10% of range).
    Returns:
        noisy_image: (C, H, W) tensor.
    """
    # Images are normalized to [-1, 1]
    noise = torch.randn_like(image) * sigma 
    noisy = torch.clamp(image + noise, -1, 1)
    return noisy
