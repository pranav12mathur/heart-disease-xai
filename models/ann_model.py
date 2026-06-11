"""
Artificial Neural Network for Heart Disease Prediction
Architecture: Deep feedforward network with BatchNorm, Dropout, and residual connections
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


class HeartDiseaseANN(nn.Module):
    """
    Deep ANN with:
    - 4 hidden layers (256 -> 128 -> 64 -> 32)
    - BatchNorm + Dropout after each layer
    - Skip connection from input to layer 3
    """

    def __init__(self, input_dim=13, dropout=0.3):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block3 = nn.Sequential(
            nn.Linear(128 + input_dim, 64),  # skip connection from input
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )

        self.output = nn.Linear(32, 1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_in = self.input_bn(x)
        h1 = self.block1(x_in)
        h2 = self.block2(h1)
        h3 = self.block3(torch.cat([h2, x_in], dim=1))  # skip
        h4 = self.block4(h3)
        return self.output(h4)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits).squeeze()
        return probs.numpy()


class ANNTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, patience=20):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.patience = patience
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _to_tensor(self, X, y=None):
        X_t = torch.FloatTensor(X)
        if y is not None:
            return X_t, torch.FloatTensor(y).unsqueeze(1)
        return X_t

    def train(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
        X_t, y_t = self._to_tensor(X_train, y_train)
        X_v, y_v = self._to_tensor(X_val, y_val)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        best_state = None
        wait = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss, train_correct = 0, 0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item() * len(xb)
                train_correct += ((torch.sigmoid(logits) > 0.5) == yb).sum().item()

            train_loss /= len(X_train)
            train_acc = train_correct / len(X_train)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_v)
                val_loss = self.criterion(val_logits, y_v).item()
                val_acc = ((torch.sigmoid(val_logits) > 0.5) == y_v).float().mean().item()

            self.scheduler.step(val_loss)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)

        return self.history

    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_t = self._to_tensor(X_test)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).squeeze().numpy()

        preds = (probs > 0.5).astype(int)
        results = {
            'accuracy': accuracy_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probs),
            'f1': f1_score(y_test, preds),
            'confusion_matrix': confusion_matrix(y_test, preds),
            'classification_report': classification_report(y_test, preds),
            'probs': probs,
            'preds': preds
        }
        return results

    def save(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model_state'])
        self.history = ckpt['history']
        print(f"Model loaded from {path}")