import torch
import torch.nn as nn
import torch.nn.functional as F
from .EnhancementModule import EnhancementModule


class U2NetE(nn.Module):
    def __init__(self, base_model, enhancement_mode="hybrid", normalize=False):
        """
        Args:
            base_model: U2NET_lite instance
            enhancement_mode: 'rgb', 'green', 'hybrid'
            normalize: whether to apply post-enhancement normalization
        """
        super().__init__()
        self.enhance = EnhancementModule(mode=enhancement_mode)
        self.u2net = base_model
        self.normalize = normalize

        if self.normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, return_enhanced=False):
        # --- enhancement ---
        x = self.enhance(x)
        enhanced = x
        # --- normalization ---
        if self.normalize:
            x = (x - self.mean) / self.std

        # --- forward through base U2NET ---
        side_outputs = self.u2net(x)  # returns [d0, d1, ..., d6]
        
        if return_enhanced:
            return side_outputs, enhanced
        return side_outputs
