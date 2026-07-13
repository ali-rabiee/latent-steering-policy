"""Quaternion / rotation-vector / rot6d utilities, pure torch, wxyz convention.

Conventions match kinova-isaac's data_collection/core/logger.py:
- quaternions are [w, x, y, z], Hamilton product
- a base-frame delta between consecutive EE orientations is the LEFT-multiplied
  error quaternion  q_err = q_curr * conj(q_prev),  rotvec = axis * angle.

Unlike the logger, `quat_to_rotvec` canonicalizes the quaternion sign (w >= 0)
so the returned rotation vector is always the shortest arc (angle <= pi).
"""

from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so w >= 0 (q and -q are the same rotation)."""
    return torch.where(q[..., 0:1] < 0.0, -q, q)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 * q2, both [w, x, y, z]."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v by quaternion(s) q."""
    qw = q[..., 0:1]
    qv = q[..., 1:4]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_to_rotvec(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Unit quaternion [w,x,y,z] -> rotation vector (axis*angle), shortest arc."""
    q = quat_canonical(quat_normalize(q))
    w = torch.clamp(q[..., 0], -1.0, 1.0)
    v = q[..., 1:4]
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    angle = 2.0 * torch.atan2(v_norm, w.unsqueeze(-1))
    axis = v / v_norm
    return axis * angle


def rotvec_to_quat(rv: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Rotation vector (axis*angle) -> unit quaternion [w,x,y,z]."""
    angle = torch.linalg.norm(rv, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(half)/angle is well-behaved near 0: ~0.5 - angle^2/48
    small = angle < eps
    sin_half_over_angle = torch.where(
        small, 0.5 * torch.ones_like(angle), torch.sin(half) / angle.clamp_min(eps)
    )
    w = torch.cos(half)
    xyz = rv * sin_half_over_angle
    return torch.cat([w, xyz], dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion [w,x,y,z] -> rotation matrix (..., 3, 3)."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(dim=-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1)
    row1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1)
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion -> 6d rotation representation (first two COLUMNS of R)."""
    m = quat_to_matrix(q)
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def delta_rotvec(q_prev: torch.Tensor, q_curr: torch.Tensor) -> torch.Tensor:
    """Base-frame delta rotation q_curr = dq * q_prev, returned as shortest-arc rotvec."""
    return quat_to_rotvec(quat_multiply(q_curr, quat_conjugate(q_prev)))


def apply_delta_rotvec(q_prev: torch.Tensor, drot: torch.Tensor) -> torch.Tensor:
    """Inverse of delta_rotvec: q_curr = quat(drot) * q_prev."""
    return quat_normalize(quat_multiply(rotvec_to_quat(drot), q_prev))
