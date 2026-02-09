from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn

from config.settings import GLOBAL_SEED


class AdversarialPerturbationGenerator:
    """Generates adversarial perturbations for neural network models.

    Implements gradient-based adversarial attack methods including BIM, PGD,
    and C&W attacks. These methods generate subtle perturbations designed to
    fool neural networks while remaining imperceptible or minimally perceptible.

    All methods support both classification and regression tasks.

    Attributes:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int = GLOBAL_SEED):
        """Initialize adversarial perturbation generator.

        Args:
            seed: Random seed for reproducibility. Default from config.
        """
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def bim(self,
            model: nn.Module,
            X: np.ndarray,
            y: np.ndarray,
            epsilon: float,
            alpha: float,
            num_iter: int,
            task_type: Literal["regression", "classification"],
            device: torch.device) -> np.ndarray:
        """Generate adversarial examples using Basic Iterative Method (BIM).

        BIM is an iterative version of FGSM that applies the fast gradient
        sign method multiple times with a smaller step size, clipping the
        result to stay within the epsilon-ball after each step.

        Args:
            model: PyTorch model to attack.
            X: Input features, shape (n_samples, n_features).
            y: True labels (classification) or values (regression), shape (n_samples,).
            epsilon: Maximum L-infinity norm of perturbation.
            alpha: Step size for each iteration.
            num_iter: Number of iterative steps.
            task_type: Type of task (classification or regression).
            device: PyTorch device for computation.

        Returns:
            Adversarial examples, shape (n_samples, n_features).
        """
        model.eval()

        X_tensor = torch.FloatTensor(X).to(device)
        X_adv = X_tensor.clone().detach()

        if task_type == "regression":
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
            criterion = nn.MSELoss()
        else:
            y_tensor = torch.LongTensor(y).to(device)
            criterion = nn.CrossEntropyLoss()

        for _ in range(num_iter):
            X_adv.requires_grad = True

            outputs = model(X_adv)
            loss = criterion(outputs, y_tensor)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                # GRADIENT SIGN METHOD STEP
                perturbation = alpha * X_adv.grad.sign()
                X_adv = X_adv + perturbation

                # PROJECT BACK TO EPSILON BALL
                eta = torch.clamp(X_adv - X_tensor, min=-epsilon, max=epsilon)
                X_adv = X_tensor + eta

        return X_adv.detach().cpu().numpy()

    def pgd(self,
            model: nn.Module,
            X: np.ndarray,
            y: np.ndarray,
            epsilon: float,
            alpha: float,
            num_iter: int,
            task_type: Literal["regression", "classification"],
            device: torch.device,
            random_start: bool = True) -> np.ndarray:
        """Generate adversarial examples using Projected Gradient Descent (PGD).

        PGD is similar to BIM but starts from a random point within the
        epsilon-ball, making it a stronger attack. Often considered the
        gold standard for adversarial robustness evaluation.

        Args:
            model: PyTorch model to attack.
            X: Input features, shape (n_samples, n_features).
            y: True labels (classification) or values (regression), shape (n_samples,).
            epsilon: Maximum L-infinity norm of perturbation.
            alpha: Step size for each iteration.
            num_iter: Number of iterative steps.
            task_type: Type of task (classification or regression).
            device: PyTorch device for computation.
            random_start: Whether to initialize from random point in epsilon-ball.
                True makes the attack stronger.

        Returns:
            Adversarial examples, shape (n_samples, n_features).
        """
        model.eval()

        X_tensor = torch.FloatTensor(X).to(device)

        # RANDOM INITIALIZATION IN EPSILON BALL
        if random_start:
            X_adv = X_tensor + torch.empty_like(X_tensor).uniform_(-epsilon, epsilon)
            X_adv = torch.clamp(X_adv, X_tensor.min().item(), X_tensor.max().item())
        else:
            X_adv = X_tensor.clone().detach()

        if task_type == "regression":
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
            criterion = nn.MSELoss()
        else:
            y_tensor = torch.LongTensor(y).to(device)
            criterion = nn.CrossEntropyLoss()

        for _ in range(num_iter):
            X_adv.requires_grad = True

            outputs = model(X_adv)
            loss = criterion(outputs, y_tensor)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                # GRADIENT ASCENT STEP
                perturbation = alpha * X_adv.grad.sign()
                X_adv = X_adv + perturbation

                # PROJECT BACK TO EPSILON BALL AROUND ORIGINAL INPUT
                eta = torch.clamp(X_adv - X_tensor, min=-epsilon, max=epsilon)
                X_adv = X_tensor + eta

        return X_adv.detach().cpu().numpy()

    def cw(self,
           model: nn.Module,
           X: np.ndarray,
           y: np.ndarray,
           c: float,
           kappa: float,
           num_iter: int,
           learning_rate: float,
           task_type: Literal["regression", "classification"],
           device: torch.device,
           targeted: bool = False) -> np.ndarray:
        """Generate adversarial examples using Carlini & Wagner (C&W) attack.

        C&W is an optimization-based attack that minimizes perturbation size
        while ensuring misclassification. Generally produces smaller
        perturbations than gradient-based methods but is more computationally
        expensive.

        Args:
            model: PyTorch model to attack.
            X: Input features, shape (n_samples, n_features).
            y: True labels (classification) or values (regression), shape (n_samples,).
            c: Confidence parameter balancing adversarial loss vs perturbation size.
                Higher values prioritize misclassification over small perturbations.
            kappa: Confidence margin for misclassification.
            num_iter: Number of optimization iterations.
            learning_rate: Learning rate for Adam optimizer.
            task_type: Type of task (classification or regression).
            device: PyTorch device for computation.
            targeted: Whether to perform targeted attack (not fully implemented).

        Returns:
            Adversarial examples, shape (n_samples, n_features).

        Note:
            The implementation uses direct perturbation optimization rather than
            the tanh space transformation from the original paper.
        """
        model.eval()

        X_tensor = torch.FloatTensor(X).to(device)

        # USE TANH SPACE FOR BOX CONSTRAINTS
        # X_adv = 0.5 * (tanh(w) + 1) ensures X_adv in [0, 1] if we normalize
        # FOR SIMPLICITY WE LL WORK DIRECTLY IN INPUT SPACE WITH CLAMPING
        w = torch.zeros_like(X_tensor, requires_grad=True, device=device)

        optimizer = torch.optim.Adam([w], lr=learning_rate)

        if task_type == "regression":
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
        else:
            y_tensor = torch.LongTensor(y).to(device)

        for _ in range(num_iter):
            X_adv = X_tensor + w

            outputs = model(X_adv)

            # L2 DISTANCE PENALTY
            l2_dist = torch.norm((X_adv - X_tensor).view(X_tensor.size(0), -1), p=2, dim=1)

            if task_type == "regression":
                # FOR REGRESSION MAXIMIZE PREDICTION ERROR
                # f(x') = ||y - model(x')||^2
                pred_loss = torch.norm(outputs - y_tensor, p=2, dim=1)

                if targeted:
                    # MINIMIZE DISTANCE TO TARGET MAKE PREDICTION CLOSER TO TARGET
                    adv_loss = pred_loss
                else:
                    # MAXIMIZE DISTANCE FROM TRUE VALUE MAKE PREDICTION WORSE
                    # SINCE OPTIMIZER MINIMIZES LOSS USE NEGATIVE TO MAXIMIZE ERROR
                    adv_loss = -pred_loss
            else:
                # FOR CLASSIFICATION MAXIMIZE LOGIT DIFFERENCE
                # f(x') = max(Z(x')_y - max_{i`y} Z(x')_i, -kappa)
                one_hot_y = torch.zeros_like(outputs)
                one_hot_y.scatter_(1, y_tensor.unsqueeze(1), 1)

                real = torch.sum(outputs * one_hot_y, dim=1)
                other = torch.max((1 - one_hot_y) * outputs - one_hot_y * 10000, dim=1)[0]

                if targeted:
                    # MAXIMIZE TARGET CLASS LOGIT
                    adv_loss = torch.clamp(other - real, min=-kappa)
                else:
                    # MINIMIZE CORRECT CLASS LOGIT
                    adv_loss = torch.clamp(real - other, min=-kappa)

            # Total loss: c * f(x') + ||x' - x||_2^2
            loss = torch.sum(c * adv_loss + l2_dist)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        X_adv = (X_tensor + w).detach()
        return X_adv.cpu().numpy()

    def perturb(self,
                model: nn.Module,
                X: np.ndarray,
                y: np.ndarray,
                method: Literal["bim", "pgd", "cw"],
                task_type: Literal["regression", "classification"],
                device: torch.device,
                epsilon: float = 0.1,
                alpha: Optional[float] = None,
                num_iter: int = 10,
                c: float = 1.0,
                kappa: float = 0.0,
                learning_rate: float = 0.01) -> np.ndarray:
        """Apply adversarial perturbation using specified method.

        Dispatcher method that routes to the appropriate adversarial attack
        based on the method parameter. Automatically sets alpha if not provided.

        Args:
            model: PyTorch model to attack.
            X: Input features, shape (n_samples, n_features).
            y: True labels (classification) or values (regression), shape (n_samples,).
            method: Attack method to use.
            task_type: Type of task (classification or regression).
            device: PyTorch device for computation.
            epsilon: Maximum perturbation magnitude for BIM/PGD (L-infinity norm).
            alpha: Step size for BIM/PGD. If None, defaults to epsilon/num_iter * 2.5.
            num_iter: Number of iterative steps.
            c: C&W confidence parameter (trade-off weight).
            kappa: C&W confidence margin.
            learning_rate: Learning rate for C&W optimizer.

        Returns:
            Adversarial examples, shape (n_samples, n_features).

        Raises:
            ValueError: If method is not recognized.
        """
        if alpha is None:
            alpha = epsilon / num_iter * 2.5

        if method == "bim":
            return self.bim(model, X, y, epsilon, alpha, num_iter, task_type, device)
        elif method == "pgd":
            return self.pgd(model, X, y, epsilon, alpha, num_iter, task_type, device)
        elif method == "cw":
            return self.cw(model, X, y, c, kappa, num_iter, learning_rate, task_type, device)
        else:
            raise ValueError(f"Unknown adversarial method: {method}")

    @staticmethod
    def make_key(method: str, epsilon: float = None, c: float = None) -> str:
        """Generate cache key for adversarial perturbation configuration.

        Args:
            method: Attack method name.
            epsilon: Perturbation budget for BIM/PGD. Only used if method is
                "bim" or "pgd".
            c: Confidence parameter for C&W. Only used if method is "cw".

        Returns:
            Cache key in format:
                - For BIM/PGD: {method}_eps{epsilon}
                - For C&W: {method}_c{c}
                - Otherwise: {method}
        """
        if method in ["bim", "pgd"]:
            return f"{method}_eps{epsilon}"
        elif method == "cw":
            return f"{method}_c{c}"
        else:
            return method
