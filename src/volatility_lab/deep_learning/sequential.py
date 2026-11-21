"""Deep learning sequential models: LSTM, GRU, TCN, and Attention architectures."""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from volatility_lab.utils.reproducibility import seed_everything
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class LSTMVolatilityModel(nn.Module):
        """Multi-layer LSTM for sequential financial volatility prediction."""

        def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus(),  # Enforce strictly positive variance
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            last_out = out[:, -1, :]
            return self.fc(last_out)


    class GRUVolatilityModel(nn.Module):
        """Multi-layer GRU for sequential volatility prediction."""

        def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.gru(x)
            last_out = out[:, -1, :]
            return self.fc(last_out)


    class TemporalConvBlock(nn.Module):
        """Causal dilated convolutional block for TCN."""

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                dilation=dilation,
            )
            self.relu = nn.ReLU()
            self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, L]
            out = self.conv(x)
            # Remove future lookahead from standard symmetric padding
            if self.padding > 0:
                out = out[:, :, :-self.padding]
            return self.relu(out + self.residual(x))


    class TCNVolatilityModel(nn.Module):
        """Temporal Convolutional Network with causal dilations."""

        def __init__(self, input_dim: int, channels: List[int] = [32, 32, 16], kernel_size: int = 3):
            super().__init__()
            layers = []
            in_ch = input_dim
            for i, out_ch in enumerate(channels):
                dilation = 2 ** i
                layers.append(TemporalConvBlock(in_ch, out_ch, kernel_size, dilation))
                in_ch = out_ch
            self.network = nn.Sequential(*layers)
            self.fc = nn.Sequential(
                nn.Linear(channels[-1], 1),
                nn.Softplus(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, C] -> permute to [B, C, L]
            x_perm = x.permute(0, 2, 1)
            feat = self.network(x_perm)  # [B, C, L]
            last_step = feat[:, :, -1]
            return self.fc(last_step)


    class AttentionVolatilityModel(nn.Module):
        """Multi-Head Self-Attention sequential architecture for volatility forecasting."""

        def __init__(self, input_dim: int, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, embed_dim)
            self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
            self.norm = nn.LayerNorm(embed_dim)
            self.fc = nn.Sequential(
                nn.Linear(embed_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            proj = self.input_proj(x)
            attn_out, _ = self.mha(proj, proj, proj)
            normed = self.norm(proj + attn_out)
            last_step = normed[:, -1, :]
            return self.fc(last_step)


class DeepVolatilityForecaster:
    """Trainer and inference orchestrator for PyTorch volatility models."""

    def __init__(
        self,
        architecture: str = "lstm",
        seq_length: int = 20,
        hidden_dim: int = 32,
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 0.001,
        patience: int = 6,
        seed: int = 42,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for DeepVolatilityForecaster.")
        self.architecture = architecture.lower()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self.model: Optional[nn.Module] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.training_history: List[Dict[str, float]] = []

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length + 1):
            X_seq.append(X[i : i + self.seq_length])
            y_seq.append(y[i + self.seq_length - 1])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, val_split: float = 0.2) -> "DeepVolatilityForecaster":
        seed_everything(self.seed)
        X_vals = X_train.values
        y_vals = y_train.values

        # Normalize features strictly based on training partition
        split_idx = int(len(X_vals) * (1.0 - val_split))
        X_tr_raw = X_vals[:split_idx]

        self.mean_ = np.nanmean(X_tr_raw, axis=0)
        self.std_ = np.nanstd(X_tr_raw, axis=0)
        self.std_[self.std_ == 0] = 1.0

        X_norm = (X_vals - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, 0.0)

        X_seq, y_seq = self._create_sequences(X_norm, y_vals)
        n_samples = len(X_seq)
        val_samples = max(int(n_samples * val_split), 1)
        train_samples = n_samples - val_samples

        X_tr_seq, y_tr_seq = X_seq[:train_samples], y_seq[:train_samples]
        X_vl_seq, y_vl_seq = X_seq[train_samples:], y_seq[train_samples:]

        input_dim = X_train.shape[1]
        if self.architecture == "lstm":
            self.model = LSTMVolatilityModel(input_dim=input_dim, hidden_dim=self.hidden_dim)
        elif self.architecture == "gru":
            self.model = GRUVolatilityModel(input_dim=input_dim, hidden_dim=self.hidden_dim)
        elif self.architecture == "tcn":
            self.model = TCNVolatilityModel(input_dim=input_dim)
        elif self.architecture == "attention":
            self.model = AttentionVolatilityModel(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown deep architecture: {self.architecture}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # Use MSE on log-transformed variance to penalize proportional error
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.training_history = []
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for b_x, b_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(b_x)
                loss = criterion(torch.log(pred + 1e-6), torch.log(b_y + 1e-6))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(b_x)
            train_loss /= len(train_dataset)

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_x = torch.from_numpy(X_vl_seq)
                val_y = torch.from_numpy(y_vl_seq).unsqueeze(1)
                val_pred = self.model(val_x)
                val_loss = criterion(torch.log(val_pred + 1e-6), torch.log(val_y + 1e-6)).item()

            self.training_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_loss:.5f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts forward variance for sequences extracted from X."""
        if self.model is None or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model must be fitted before predict.")
        self.model.eval()

        X_norm = (X.values - self.mean_) / self.std_
        X_norm = np.nan_to_num(X_norm, 0.0)

        if len(X_norm) < self.seq_length:
            # Replicate leading rows to form at least one sequence
            pad_len = self.seq_length - len(X_norm)
            pad = np.tile(X_norm[0], (pad_len, 1))
            X_norm = np.vstack([pad, X_norm])

        X_seq = []
        for i in range(len(X_norm) - self.seq_length + 1):
            X_seq.append(X_norm[i : i + self.seq_length])
        X_tensor = torch.from_numpy(np.array(X_seq, dtype=np.float32))

        with torch.no_grad():
            preds = self.model(X_tensor).squeeze(-1).numpy()
        return np.maximum(preds, 1e-6)
