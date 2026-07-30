"""Vision encoder: ResNet-18 (GroupNorm) + SpatialSoftmax, per Diffusion Policy.

No pretrained weights, BatchNorm replaced with GroupNorm (EMA-friendly, small
batch friendly), global average pooling replaced with SpatialSoftmax keypoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from lsteer.data import schema


def _replace_bn_with_gn(module: nn.Module, features_per_group: int = 16) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = max(1, num_channels // features_per_group)
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))
        else:
            _replace_bn_with_gn(child, features_per_group)
    return module


class SpatialSoftmax(nn.Module):
    """Expected 2D keypoint locations over feature-map attention (Levine et al.)."""

    def __init__(self, in_channels: int, num_keypoints: int = 32, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.temperature = temperature
        self.num_keypoints = num_keypoints
        self._grid_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _grid(self, h: int, w: int, device, dtype):
        key = (h, w)
        if key not in self._grid_cache:
            ys = torch.linspace(-1.0, 1.0, h)
            xs = torch.linspace(-1.0, 1.0, w)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self._grid_cache[key] = (gx.reshape(1, 1, -1), gy.reshape(1, 1, -1))
        gx, gy = self._grid_cache[key]
        return gx.to(device=device, dtype=dtype), gy.to(device=device, dtype=dtype)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feat.shape
        att = self.proj(feat).reshape(b, self.num_keypoints, h * w)
        att = torch.softmax(att / self.temperature, dim=-1)
        gx, gy = self._grid(h, w, feat.device, feat.dtype)
        ex = (att * gx).sum(dim=-1)
        ey = (att * gy).sum(dim=-1)
        return torch.cat([ex, ey], dim=-1)  # (B, 2*num_keypoints)


class VisionEncoder(nn.Module):
    def __init__(self, num_keypoints: int = 32, out_dim: int = 128):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        _replace_bn_with_gn(resnet)
        # keep everything up to layer4; drop avgpool + fc
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.spatial_softmax = SpatialSoftmax(512, num_keypoints)
        self.head = nn.Linear(2 * num_keypoints, out_dim)
        self.out_dim = out_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) in [0, 1] -> (B, out_dim)."""
        feat = self.backbone(img)
        kp = self.spatial_softmax(feat)
        return self.head(kp)


class ObsEncoder(nn.Module):
    """Encodes a multi-camera observation window into one global conditioning
    vector.

    obs: {img_<cam>: (B,T_o,3,H,W) for each camera, state: (B,T_o,D_s)}
      -> (B, T_o*(img_dim*n_cams + D_s))

    Each camera gets its OWN VisionEncoder (no weight sharing): the front and
    wrist views have completely different statistics and a shared trunk would
    have to straddle both. Per-camera features are concatenated in the fixed
    `camera_names` order, then concatenated with the state, per timestep.
    """

    def __init__(
        self,
        state_dim: int,
        obs_horizon: int,
        img_dim: int = 128,
        num_keypoints: int = 32,
        camera_names=schema.CAMERA_NAMES,
    ):
        super().__init__()
        self.camera_names = tuple(camera_names)
        self.vision = nn.ModuleDict(
            {cam: VisionEncoder(num_keypoints=num_keypoints, out_dim=img_dim) for cam in self.camera_names}
        )
        self.state_dim = state_dim
        self.obs_horizon = obs_horizon
        self.out_dim = obs_horizon * (img_dim * len(self.camera_names) + state_dim)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state = obs["state"]  # (B, T_o, D_s)
        feats = []
        for cam in self.camera_names:
            key = schema.camera_obs_key(cam)
            if key not in obs:
                raise KeyError(f"observation is missing '{key}' (cameras: {self.camera_names})")
            img = obs[key]  # (B, T_o, 3, H, W)
            b, t = img.shape[:2]
            feats.append(self.vision[cam](img.flatten(0, 1)).reshape(b, t, -1))
        return torch.cat(feats + [state], dim=-1).flatten(1)
