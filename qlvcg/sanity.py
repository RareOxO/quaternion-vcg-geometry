"""Numerical sanity checks for the quaternion path (report template, section 9).

Two parts: constructed edge cases whose correct answer is known, and the same
quantities measured on real training-fold records, so a report can state what the
features actually look like rather than what they should.
"""

import math

import torch

from qdg.data import PTBXLDataset

from .models import standardize
from .quaternion_utils import (
    enforce_sign_continuity,
    quaternion_angle,
    quaternion_to_rotation_matrix,
    rotate_vector_by_quaternion,
    valid_rotation_mask,
    vectors_to_quaternion,
)


def _edge_cases():
    unit = torch.tensor([[0.3, -0.5, 0.8]])
    unit = unit / unit.norm()
    parallel = vectors_to_quaternion(unit, 2.5 * unit)
    anti = vectors_to_quaternion(unit, -unit)
    zero = vectors_to_quaternion(torch.zeros(1, 3), unit)
    tiny = vectors_to_quaternion(torch.full((1, 3), 1e-12), torch.full((1, 3), -1e-12))
    q = torch.nn.functional.normalize(
        torch.randn(10_000, 4, generator=torch.manual_seed(0)), dim=-1
    )
    matrix = quaternion_to_rotation_matrix(q)
    eye = torch.eye(3).expand_as(matrix)
    return {
        "parallel_angle_deg": math.degrees(quaternion_angle(parallel).item()),
        "antiparallel_angle_deg": math.degrees(quaternion_angle(anti).item()),
        "antiparallel_rotates_u_onto_minus_u": bool(
            torch.allclose(rotate_vector_by_quaternion(unit, anti), -unit, atol=1e-6)
        ),
        "zero_vector_finite": bool(torch.isfinite(zero).all()),
        "near_zero_pair_finite": bool(torch.isfinite(tiny).all()),
        "random_q_max_orthogonality_error": (matrix.transpose(-1, -2) @ matrix - eye)
        .abs()
        .amax()
        .item(),
        "random_q_max_abs_det_minus_1": (torch.linalg.det(matrix) - 1).abs().max().item(),
        "q_and_minus_q_same_matrix": bool(
            torch.allclose(matrix, quaternion_to_rotation_matrix(-q), atol=1e-6)
        ),
    }


def _percentiles(values, points=(0.5, 0.9, 0.99)):
    return {f"p{int(p * 100)}": round(torch.quantile(values, p).item(), 4) for p in points}


def run(config, records=256):
    cache, lvcg = config["data"]["cache"], config["model"]["lvcg"]
    settings = config["model"]["qdf_lvcg"]
    dataset = PTBXLDataset(cache, "train", records, config["training"]["seed"])
    ecg = standardize(torch.stack([dataset[i]["ecg"] for i in range(len(dataset))]))
    from .models import build_model

    model = build_model(config, "V1")
    with torch.no_grad():
        p = model.vcg(ecg).transpose(1, 2)
        mask = valid_rotation_mask(p, 1, settings["min_magnitude_fraction"])
        raw = vectors_to_quaternion(p[:, :-1], p[:, 1:])
        flipped = enforce_sign_continuity(raw)
        theta = quaternion_angle(raw)
        dt = 1.0 / lvcg["fs"]
        features, _ = model.dynamics(p.transpose(1, 2))
        v2 = build_model(config, "V2")
        vcg = v2.vcg(ecg)
        backbone = v2.backbone
        beats, rr, beat_mask = backbone.beat_segmenter(vcg, ecg, rr_lead_idx=backbone.rr_lead_idx)
        beat_features, beat_valid = v2.beat_dynamics(beats, rr, beat_mask, vcg)
        steps = beats.shape[-1] - 1
        beat_dt_ms = ((rr[beat_mask] - 1).clamp_min(1.0) / (steps * lvcg["fs"])) * 1000
        # omega is channel 5 of (q, theta, omega, mask) for the V2 feature set.
        beat_omega = beat_features[:, :, 5][beat_valid]
    consecutive = (flipped[:, 1:] * flipped[:, :-1]).sum(-1)
    return {
        "edge_cases": _edge_cases(),
        "real_records": {
            "records": len(dataset),
            "dt_seconds": dt,
            "quaternion_shape": list(raw.shape),
            "max_abs_norm_minus_1": (raw.norm(dim=-1) - 1).abs().max().item(),
            "nonfinite_quaternions": int((~torch.isfinite(raw)).sum()),
            "nonfinite_features": int((~torch.isfinite(features)).sum()),
            "valid_transition_fraction": round(mask.float().mean().item(), 4),
            "min_consecutive_dot_after_sign_continuity": consecutive.min().item(),
            # A flip event changes the sign of every later quaternion in that record, so
            # the share of samples carried in the flipped sign is reported separately.
            "sign_flip_events_per_record": round(
                ((raw[:, 1:] * raw[:, :-1]).sum(-1) < 0).float().sum(-1).mean().item(), 2
            ),
            "fraction_of_samples_in_flipped_sign": round(
                ((raw * flipped).sum(-1) < 0).float().mean().item(), 4
            ),
            "theta_deg_valid": _percentiles(torch.rad2deg(theta[mask])),
            "omega_rad_per_s_valid": _percentiles(theta[mask] / dt),
        },
        "beat_level_v2": {
            "beats_shape": list(beats.shape),
            "real_beats_per_record": _percentiles(beat_mask.float().sum(-1)),
            "patch_step_dt_ms": _percentiles(beat_dt_ms),
            "valid_transition_fraction_in_real_beats": round(
                (beat_valid.sum() / (beat_mask.sum() * steps)).item(), 4
            ),
            "nonfinite_features": int((~torch.isfinite(beat_features)).sum()),
            "omega_rad_per_s_valid": _percentiles(beat_omega),
        },
    }
