import torch

from lsteer.utils.rotations import (
    apply_delta_rotvec,
    delta_rotvec,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    quat_to_rot6d,
    quat_to_rotvec,
    rotvec_to_quat,
)


def random_quats(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return quat_normalize(torch.randn(n, 4, generator=g, dtype=torch.float64))


def test_rotvec_quat_roundtrip():
    g = torch.Generator().manual_seed(1)
    rv = torch.randn(100, 3, generator=g, dtype=torch.float64)
    rv = rv / rv.norm(dim=-1, keepdim=True) * torch.rand(100, 1, generator=g).double() * 3.0
    rv2 = quat_to_rotvec(rotvec_to_quat(rv))
    assert torch.allclose(rv, rv2, atol=1e-8)


def test_rotvec_shortest_arc_under_sign_flip():
    q = random_quats(50)
    assert torch.allclose(quat_to_rotvec(q), quat_to_rotvec(-q), atol=1e-8)
    assert (quat_to_rotvec(q).norm(dim=-1) <= torch.pi + 1e-9).all()


def test_matrix_orthonormal():
    m = quat_to_matrix(random_quats(50))
    eye = torch.eye(3, dtype=torch.float64).expand(50, 3, 3)
    assert torch.allclose(m @ m.transpose(-1, -2), eye, atol=1e-9)
    assert torch.allclose(torch.linalg.det(m), torch.ones(50, dtype=torch.float64), atol=1e-9)


def test_rot6d_matches_matrix_columns():
    q = random_quats(10)
    m = quat_to_matrix(q)
    r6 = quat_to_rot6d(q)
    assert torch.allclose(r6[:, 0:3], m[:, :, 0], atol=1e-12)
    assert torch.allclose(r6[:, 3:6], m[:, :, 1], atol=1e-12)


def test_delta_rotvec_roundtrip():
    """Recovered delta re-applied to q_prev must give q_curr (logger convention:
    q_curr = dq * q_prev, left-multiplied base-frame delta)."""
    q_prev = random_quats(50, seed=2)
    q_curr = random_quats(50, seed=3)
    drot = delta_rotvec(q_prev, q_curr)
    q_rec = apply_delta_rotvec(q_prev, drot)
    dot = (q_rec * q_curr).sum(dim=-1).abs()  # q and -q are the same rotation
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-8)


def test_delta_rotvec_small_angle():
    q_prev = random_quats(20, seed=4)
    small = torch.randn(20, 3, dtype=torch.float64) * 1e-5
    q_curr = quat_multiply(rotvec_to_quat(small), q_prev)
    drot = delta_rotvec(q_prev, q_curr)
    assert torch.allclose(drot, small, atol=1e-9)
