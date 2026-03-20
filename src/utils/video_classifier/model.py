"""
Video dynamics classifier: pretrained ResNet18 backbone, one feature vector per frame, mean over time, linear to 6 classes.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class VideoDynamicsClassifier(nn.Module):
    """Encode each of T frames with ResNet18, mean-pool over time, linear to num_classes."""

    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone_epochs: int = 0):
        super().__init__()
        self.num_classes = num_classes
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()  # remove classifier; output 512-d per frame
        self.feat_dim = 512
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self._freeze_epochs = freeze_backbone_epochs
        self._frozen = freeze_backbone_epochs > 0
        if self._frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        if self._frozen:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self._frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        # ImageNet normalize
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std.clamp(min=1e-6)
        feats = self.backbone(x)  # (B*T, 512)
        feats = feats.view(B, T, -1).mean(dim=1)  # (B, 512)
        return self.classifier(feats)  # (B, num_classes)
