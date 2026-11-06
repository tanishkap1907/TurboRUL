"""
train_lstm.py — Train an LSTM model for sequence-based RUL prediction.

Why LSTM? Sensor degradation is a temporal process — an engine doesn't
just have bad readings at one moment; it *trends* toward failure over
many cycles. LSTMs learn these sequential patterns directly from raw
time-series, without hand-crafted rolling features.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    """
    LSTM network for RUL regression.

    Architecture:
      - 2-layer LSTM with dropout
      - Fully connected head: hidden → 64 → 1
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the output at the last timestep
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq,
               epochs: int = 50, batch_size: int = 256, lr: float = 1e-3,
               save_dir: str = "models"):
    """
    Train LSTM model with early stopping.

    Returns
    -------
    model, metrics dict, training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_seq).to(device)
    y_train_t = torch.FloatTensor(y_train_seq).to(device)
    X_test_t = torch.FloatTensor(X_test_seq).to(device)
    y_test_t = torch.FloatTensor(y_test_seq).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    input_dim = X_train_seq.shape[2]
    model = LSTMModel(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # Training loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n--- Training LSTM ({epochs} epochs) ---")
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t).item()

        train_loss = np.mean(train_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "lstm_rul.pt"))
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(os.path.join(save_dir, "lstm_rul.pt"),
                                     weights_only=True))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()
    y_true = y_test_seq

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    # PHM score
    d = y_pred - y_true
    phm = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

    metrics = {"rmse": round(float(rmse), 2), "mae": round(float(mae), 2),
               "r2": round(float(r2), 4), "phm_score": round(float(phm), 2)}

    print(f"\nLSTM Results:")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v}")

    return model, metrics, history
