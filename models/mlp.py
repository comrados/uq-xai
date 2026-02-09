from typing import Optional, Literal, List, Self
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from abc import ABC, abstractmethod

from config.settings import GLOBAL_SEED, MLP_DEFAULTS
from models.base import BaseModel


def get_device() -> torch.device:
    """Determine the best available device for PyTorch operations.

    Checks for CUDA GPU, then Apple Metal Performance Shaders (MPS),
    falling back to CPU if neither is available.

    Returns:
        torch.device: The device to use for tensor operations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPNetwork(nn.Module):
    """Multi-layer perceptron neural network architecture.

    A feedforward neural network with configurable hidden layers, batch
    normalization, ReLU activation, and dropout regularization.

    Attributes:
        network: Sequential container of network layers.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout: float):
        """Initialize MLP architecture.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Number of output units (classes or regression outputs).
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor, shape (batch_size, input_dim).

        Returns:
            Output tensor, shape (batch_size, output_dim).
        """
        return self.network(x)


class BaseMLPModel(BaseModel, ABC):
    """Abstract base class for MLP models with shared training logic.

    Provides common training functionality including early stopping, validation,
    progress tracking, and device management. Subclasses must implement
    task-specific methods for output dimension, loss criterion, and target
    preparation.

    Attributes:
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability for regularization.
        learning_rate: Learning rate for Adam optimizer.
        batch_size: Batch size for mini-batch training.
        max_epochs: Maximum number of training epochs.
        early_stopping_patience: Epochs to wait before early stopping.
        device: PyTorch device for tensor operations.
        model: The MLPNetwork instance.
        best_state: Best model state dict during training.
    """

    def __init__(self,
                 hidden_dims: List[int] = MLP_DEFAULTS['hidden_dims'],
                 dropout: float = MLP_DEFAULTS['dropout'],
                 learning_rate: float = MLP_DEFAULTS['learning_rate'],
                 batch_size: int = MLP_DEFAULTS['batch_size'],
                 max_epochs: int = MLP_DEFAULTS['max_epochs'],
                 early_stopping_patience: int = MLP_DEFAULTS['early_stopping_patience'],
                 seed: int = GLOBAL_SEED):
        """Initialize base MLP model.

        Args:
            hidden_dims: List of hidden layer dimensions. Default from config.
            dropout: Dropout probability between 0 and 1. Default from config.
            learning_rate: Learning rate for Adam optimizer. Default from config.
            batch_size: Number of samples per training batch. Default from config.
            max_epochs: Maximum training epochs. Default from config.
            early_stopping_patience: Number of epochs without improvement before
                stopping. Default from config.
            seed: Random seed for reproducibility. Default from config.
        """
        super().__init__(seed=seed)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = get_device()
        self.model = None
        self.best_state = None
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @property
    def name(self) -> str:
        """Generate a unique identifier for this model configuration.

        Returns:
            String in format: {task_type}_mlp_{hidden_dims}_d{dropout}.
        """
        dims_str = "_".join(map(str, self.hidden_dims))
        return f"{self.task_type}_mlp_{dims_str}_d{self.dropout}"

    @property
    def supports_gradients(self) -> bool:
        """Indicate that this model supports gradient-based operations.

        Returns:
            True, as MLP models support gradient computation.
        """
        return True

    @abstractmethod
    def _get_output_dim(self, y_train: np.ndarray) -> int:
        """Determine the output dimension from training labels.

        Args:
            y_train: Training labels, shape (n_samples,).

        Returns:
            Number of output units required.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_criterion(self) -> nn.Module:
        """Get the loss criterion for this task.

        Returns:
            PyTorch loss module appropriate for the task.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_target(self, y: np.ndarray) -> torch.Tensor:
        """Convert target labels to appropriate tensor format.

        Args:
            y: Target labels, shape (n_samples,).

        Returns:
            PyTorch tensor in the format expected by the loss criterion.
        """
        raise NotImplementedError
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Self:
        """Train the MLP model with early stopping support.

        Trains the model using mini-batch gradient descent with Adam optimizer.
        If validation data is provided, performs early stopping based on
        validation loss.

        Args:
            X_train: Training features, shape (n_samples, n_features).
            y_train: Training labels, shape (n_samples,).
            X_val: Validation features, shape (n_val_samples, n_features).
                Optional, enables early stopping if provided.
            y_val: Validation labels, shape (n_val_samples,).
                Optional, enables early stopping if provided.

        Returns:
            Self for method chaining.
        """
        
        input_dim = X_train.shape[1]
        output_dim = self._get_output_dim(y_train)
        
        self.model = MLPNetwork(input_dim, self.hidden_dims, output_dim, self.dropout)
        self.model.to(self.device)
        
        criterion = self._get_criterion()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), self._prepare_target(y_train)),
            batch_size=self.batch_size, shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), self._prepare_target(y_val)),
                batch_size=self.batch_size, shuffle=False
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        pbar = tqdm(range(self.max_epochs), desc=f"Training {self.name}")
        
        for epoch in pbar:
            # TRAIN
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # VALIDATION EARLY STOPPING
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        val_loss += criterion(self.model(X_batch), y_batch).item()
                val_loss /= len(val_loader)
                
                pbar.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        self.model.load_state_dict(self.best_state)
                        break
            else:
                pbar.set_postfix({"train": f"{train_loss:.4f}"})
        
        self.is_fitted = True
        return self


class MLPClassifier(BaseMLPModel):
    """Multi-layer perceptron for classification tasks.

    Uses cross-entropy loss and softmax activation for multi-class
    classification. Supports gradient-based uncertainty quantification
    and adversarial perturbations.

    Attributes:
        task_type: Classification task identifier.
        num_classes: Number of classes in the classification problem.
    """

    task_type: Literal["classification"] = "classification"

    def _get_output_dim(self, y_train: np.ndarray) -> int:
        """Determine number of classes from training labels.

        Args:
            y_train: Training labels, shape (n_samples,).

        Returns:
            Number of unique classes.
        """
        self.num_classes = len(np.unique(y_train))
        return self.num_classes

    def _get_criterion(self) -> nn.Module:
        """Get cross-entropy loss for classification.

        Returns:
            CrossEntropyLoss module.
        """
        return nn.CrossEntropyLoss()

    def _prepare_target(self, y: np.ndarray) -> torch.Tensor:
        """Convert labels to LongTensor for cross-entropy loss.

        Args:
            y: Target labels, shape (n_samples,).

        Returns:
            LongTensor of shape (n_samples,).
        """
        return torch.LongTensor(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predicted class labels, shape (n_samples,).
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.FloatTensor(X).to(self.device))
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Class probabilities, shape (n_samples, n_classes).
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.FloatTensor(X).to(self.device))
            proba = torch.softmax(outputs, dim=1)
        return proba.cpu().numpy()