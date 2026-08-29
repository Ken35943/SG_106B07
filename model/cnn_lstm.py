"""
CSI Fall Detection CNN-BiLSTM-Attention Model.

Hybrid architecture combining 1D-CNN feature extraction, bidirectional LSTM
temporal modelling, and multi-head self-attention for CSI-based fall detection
on ESP32-S3 captured WiFi channel state information.

Input shape : (batch, time_steps=100, features=20)
Output shape: (batch, num_classes=2)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Multi-head self-attention for temporal weighting.

    Computes scaled dot-product attention across the temporal dimension,
    then aggregates the attended representations into a fixed-size vector.

    Args:
        embed_dim: Dimensionality of input features.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights.

    Input shape:  (batch, seq_len, embed_dim)
    Output shape: (batch, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-head self-attention + temporal pooling.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Attended output of shape (batch, embed_dim).
        """
        batch_size, seq_len, _ = x.shape
        # x: (batch, seq_len, embed_dim)

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, embed_dim)
        K = self.W_k(x)  # (batch, seq_len, embed_dim)
        V = self.W_v(x)  # (batch, seq_len, embed_dim)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # attn_scores: (batch, num_heads, seq_len, seq_len)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # attn_output: (batch, seq_len, embed_dim)

        attn_output = self.W_o(attn_output)
        attn_output = self.layer_norm(attn_output + x)  # Residual connection
        # attn_output: (batch, seq_len, embed_dim)

        # Temporal mean pooling to collapse sequence dimension
        output = attn_output.mean(dim=1)
        # output: (batch, embed_dim)

        return output


class CSIFallDetector(nn.Module):
    """Hybrid CNN-BiLSTM-Attention model for CSI-based fall detection.

    Architecture:
        1. Conv1D Block 1: 64 filters, kernel_size=3, ReLU, BatchNorm
        2. Conv1D Block 2: 128 filters, kernel_size=3, ReLU, BatchNorm, MaxPool1d(2)
        3. Dropout(0.2)
        4. Bidirectional LSTM: 128 hidden units, 2 layers, dropout=0.3
        5. Multi-head self-attention: 4 heads
        6. FC: 64 → num_classes

    Args:
        input_features: Number of input features (PCA components). Default 20.
        num_classes: Number of output classes. Default 2 (fall / non-fall).
        lstm_hidden: LSTM hidden size per direction. Default 128.
        lstm_layers: Number of stacked LSTM layers. Default 2.
        lstm_dropout: Dropout between LSTM layers. Default 0.3.
        attention_heads: Number of attention heads. Default 4.
        attention_dropout: Dropout in attention layer. Default 0.1.
        cnn_dropout: Dropout after CNN blocks. Default 0.2.
    """

    def __init__(
        self,
        input_features: int = 20,
        num_classes: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        cnn_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_features = input_features
        self.num_classes = num_classes
        self.lstm_hidden = lstm_hidden

        # --- Conv1D Block 1: 64 filters, k=3, ReLU, BatchNorm ---
        self.conv1 = nn.Conv1d(
            in_channels=input_features,
            out_channels=64,
            kernel_size=3,
            padding=1,  # same padding
        )
        self.bn1 = nn.BatchNorm1d(64)

        # --- Conv1D Block 2: 128 filters, k=3, ReLU, BatchNorm, MaxPool ---
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # --- Dropout after CNN ---
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # BiLSTM output dim = 2 * lstm_hidden
        lstm_out_dim = 2 * lstm_hidden

        # --- Multi-head self-attention ---
        self.attention = AttentionLayer(
            embed_dim=lstm_out_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )

        # --- Fully connected classifier ---
        self.fc1 = nn.Linear(lstm_out_dim, 64)
        self.fc_dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

        # Weight initialization
        self._init_weights()

        # Log architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "CSIFallDetector created: %d total params (%d trainable)",
            total_params,
            trainable_params,
        )

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming (He) for Conv/Linear, orthogonal for LSTM."""
        for name, param in self.named_parameters():
            if "conv" in name and "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "lstm" in name and "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "lstm" in name and "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "fc" in name and "weight" in name:
                nn.init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                if param.dim() > 0:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN-BiLSTM-Attention network.

        Args:
            x: Input tensor of shape (batch, time_steps, features).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # x: (batch, time_steps=100, features=20)

        # --- Conv1D expects (batch, channels, seq_len) ---
        x = x.permute(0, 2, 1)
        # x: (batch, features=20, time_steps=100)

        # --- Conv1D Block 1 ---
        x = F.relu(self.bn1(self.conv1(x)))
        # x: (batch, 64, 100)

        # --- Conv1D Block 2 ---
        x = F.relu(self.bn2(self.conv2(x)))
        # x: (batch, 128, 100)

        x = self.pool(x)
        # x: (batch, 128, 50)  — MaxPool1d(2) halves temporal dim

        x = self.cnn_dropout(x)
        # x: (batch, 128, 50)

        # --- Permute back for LSTM (batch, seq_len, features) ---
        x = x.permute(0, 2, 1)
        # x: (batch, 50, 128)

        # --- Bidirectional LSTM ---
        x, _ = self.lstm(x)
        # x: (batch, 50, 256)  — 2 * lstm_hidden for bidirectional

        # --- Multi-head self-attention + temporal pooling ---
        x = self.attention(x)
        # x: (batch, 256)

        # --- Fully connected classifier ---
        x = F.relu(self.fc1(x))
        # x: (batch, 64)

        x = self.fc_dropout(x)
        x = self.fc2(x)
        # x: (batch, num_classes=2)

        return x


def get_model_summary(
    model: Optional[nn.Module] = None,
    input_features: int = 20,
    time_steps: int = 100,
    batch_size: int = 1,
) -> str:
    """Generate a human-readable model summary with layer shapes and param counts.

    Args:
        model: Model instance. If None, creates a default CSIFallDetector.
        input_features: Number of input features (for dummy forward pass).
        time_steps: Number of time steps (for dummy forward pass).
        batch_size: Batch size for the dummy input.

    Returns:
        Formatted summary string.
    """
    if model is None:
        model = CSIFallDetector(input_features=input_features)

    device = next(model.parameters()).device

    lines = []
    lines.append("=" * 72)
    lines.append("CSI Fall Detector — Model Summary")
    lines.append("=" * 72)

    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if name == "":
            continue
        # Only show leaf modules (no children of their own beyond parameters)
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if params > 0:
            lines.append(f"  {name:<40s} params: {params:>10,}  (trainable: {trainable:>10,})")
            total_params += params
            trainable_params += trainable

    lines.append("-" * 72)
    lines.append(f"  {'Total':<40s} params: {total_params:>10,}  (trainable: {trainable_params:>10,})")
    lines.append("=" * 72)

    # Dummy forward pass to verify shapes
    dummy_input = torch.randn(batch_size, time_steps, input_features, device=device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    lines.append(f"  Input shape : {tuple(dummy_input.shape)}")
    lines.append(f"  Output shape: {tuple(output.shape)}")
    lines.append("=" * 72)

    summary = "\n".join(lines)
    return summary


# ------------------------------------------------------------------
# Standalone usage
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model = CSIFallDetector(input_features=20, num_classes=2)
    summary = get_model_summary(model)
    print(summary)

    # Quick sanity check with a random batch
    dummy = torch.randn(8, 100, 20)
    output = model(dummy)
    logger.info("Forward pass OK — output: %s", output.shape)
