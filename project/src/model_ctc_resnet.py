# src/model_ctc_resnet.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

# --- VOCAB / mappings (ensure consistent lowercase mapping) ---
# adjust this vocabulary to match your dataset needs (we lowercase labels in dataset)
VOCAB = list("abcdefghijklmnopqrstuvwxyz0123456789.,!?'-")
char2idx = {c: i for i, c in enumerate(VOCAB)}
idx2char = {i: c for i, c in enumerate(VOCAB)}

class ResNetCRNN(nn.Module):
    def __init__(self, num_classes, backbone_name='resnet18', pretrained=True, hidden_size=256, n_rnn_layers=2):
        super().__init__()

        if backbone_name == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
            feat_dim = 512
        elif backbone_name == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet34(weights=weights)
            feat_dim = 512
        else:
            raise ValueError("choose resnet18 or resnet34")

        # adapt first conv for single-channel input
        w = backbone.conv1.weight.data
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if w.shape[1] == 3:
            backbone.conv1.weight.data.copy_(w.mean(dim=1, keepdim=True))

        # keep convolutional part, drop fc & avgpool
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.feat_dim = feat_dim
        self.pool_h = nn.AdaptiveAvgPool2d((1, None))   # collapse height -> 1

        self.rnn = nn.LSTM(input_size=feat_dim, hidden_size=hidden_size, num_layers=n_rnn_layers,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        feat = self.backbone(x)           # [B, C, Hf, Wf]
        feat = self.pool_h(feat)          # [B, C, 1, Wf]
        b, c, h, w = feat.size()
        feat = feat.squeeze(2)            # [B, C, Wf]
        feat = feat.permute(0, 2, 1)      # [B, Wf, C]
        rnn_out, _ = self.rnn(feat)       # [B, Wf, hidden*2]
        logits = self.fc(rnn_out)         # [B, Wf, num_classes+1]
        return logits.permute(1, 0, 2)    # [T=Wf, B, C]
