import torch
import torch.nn as nn

from .common import Classifier
from torch.nn.functional import cross_entropy


class VitMetaEncoder(nn.Module):
    def __init__(
            self,
            input_size: int = 32,
            input_channels: int = 3,
            patch_size: int = 2,
            token_dim: int = 512,
            num_classes: int = 100,
            num_layers: int = 6,
            num_heads: int = 8,
            dim_ffn: int = 2048,
            dropout: float = 0.1,
    ):
        super(VitMetaEncoder, self).__init__()

        self.num_tokens = int(input_size / patch_size) * int(input_size / patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim), requires_grad=True)
        self.patch_embedding = nn.Conv2d(input_channels, token_dim, kernel_size=patch_size, stride=patch_size)
        self.pe = nn.Parameter(torch.randn(1, self.num_tokens, token_dim), requires_grad=True)
        self.trans_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.trans_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_ffn,
                    dropout=dropout,
                    batch_first=True,
                )
            )
        self.classifier = Classifier(input_dim=token_dim, hidden_dim=token_dim*2, num_classes=num_classes)

    def forward(self, x):
        x = self.patch_embedding(x) # [B, 3, H*W]
        x = x.flatten(2).transpose(1, 2) # [B, H*W, d]
        x = x + self.pe
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) # [B, H*W+1, d]
        for layer in self.trans_layers:
            x = layer(x)
        cls_token = x[:, 0, :]
        prob_data = self.classifier(cls_token)

        return prob_data


def compute_loss(prob_data, labels):
    """
    prob_data: torch.Tensor of shape [B, C]
    labels: torch.Tensor of shape [B], dtype=torch.long
    """
    return cross_entropy(prob_data, labels, label_smoothing=0.2)