"""
Simulated Hospital Client Module for Federated Learning (Phase 3: 10-Hospital IID Baseline).
Primary Dataset: PneumoniaMNIST (MedMNIST v3).

Each simulated hospital client:
- Maintains its own local instance of SmallCNN.
- Trains strictly on its private, mutually exclusive local patient partition.
- Receives global weights from the FedAvg server, trains locally for K epochs,
  and submits updated weights and sample counts back to the server.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from federated_learning.model import SmallCNN, get_model


def get_parameters_from_model(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model weights as a list of independent NumPy arrays from state_dict.
    Includes weights, biases, and BatchNorm running statistics so
    the global aggregation maintains statistical consistency.
    Uses .copy() to prevent in-place memory sharing between PyTorch tensors and returned arrays.
    """
    return [val.detach().cpu().numpy().copy() for _, val in model.state_dict().items()]


def set_parameters_to_model(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load a list of NumPy arrays into the model state_dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


class HospitalClient(fl.client.NumPyClient):
    """
    Simulated Hospital Node participating in multi-hospital Federated Learning.
    Trains SmallCNN on local pediatric chest X-ray data.
    """

    def __init__(
        self,
        client_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        local_epochs: int = 2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: torch.device | None = None,
    ) -> None:
        self.client_id = str(client_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Independent local model instance
        self.model = get_model(in_channels=1, num_classes=2, dropout_rate=0.2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return get_parameters_from_model(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train local model using global parameters for `local_epochs`.
        """
        set_parameters_to_model(self.model, parameters)

        # Allow override of local_epochs or lr from server config
        epochs = int(config.get("local_epochs", self.local_epochs))
        lr = float(config.get("lr", self.learning_rate))

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for _ in range(epochs):
            for images, targets in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.squeeze(-1).long().to(self.device, non_blocking=True)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / max(total_samples, 1)
        train_acc = (total_correct / max(total_samples, 1)) * 100.0

        updated_params = get_parameters_from_model(self.model)
        num_examples = len(self.train_loader.dataset)

        metrics = {
            "client_id": self.client_id,
            "train_loss": round(avg_loss, 4),
            "train_accuracy": round(train_acc, 2),
            "samples": num_examples,
        }

        return updated_params, num_examples, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on local evaluation data (or local train data if val not set).
        """
        set_parameters_to_model(self.model, parameters)
        loader = self.val_loader or self.train_loader

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.squeeze(-1).long().to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        eval_loss = running_loss / max(total, 1)
        eval_acc = (correct / max(total, 1)) * 100.0

        return eval_loss, total, {"accuracy": eval_acc, "client_id": self.client_id}
