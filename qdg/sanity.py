"""The theoretical sanity checks of 方案 §6, printed as expected vs actual.

`sanity_checks()` is pure synthetic algebra and must pass before PTB-XL is touched.
`check_geometry()` is the numerical check on real cached waveforms: it reports how far
||q_first|| drifts from 1 rather than silently renormalizing (方案 §4.3).
"""

import json
import math

import torch

from .geometry import (
    first_order,
    hamilton_product,
    pure_quaternion_product,
    relation_descriptor,
    relation_norm_error,
    second_order,
    unit_direction,
)

TOLERANCE = 1e-5


def _summary(q):
    """A single quaternion prints in full; a constant sequence prints as one row."""
    q = q.reshape(-1, 4)
    row = [round(value, 6) for value in q[0].tolist()]
    if len(q) == 1:
        return str(row)
    spread = (q - q[0]).abs().max().item()
    if spread <= TOLERANCE:
        return f"{row} repeated for all {len(q)} steps"
    return f"{row} ... (spread over {len(q)} steps: {spread:.3g})"


def _check(name, expected, actual, results, tolerance=TOLERANCE):
    expected, actual = torch.as_tensor(expected).float(), torch.as_tensor(actual).float()
    error = (expected - actual).abs().max().item()
    results.append(
        {
            "test": name,
            "expected": _summary(expected),
            "actual": _summary(actual),
            "max_abs_error": error,
            "passed": error <= tolerance,
        }
    )


def _spiral(theta):
    """Unit directions rotating in the XY plane by the given angles."""
    return torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), dim=-1)


def sanity_checks():
    results = []
    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0])

    # Hamilton basics: i (x) j = k.
    i = torch.tensor([0.0, 1.0, 0.0, 0.0])
    j = torch.tensor([0.0, 0.0, 1.0, 0.0])
    _check("hamilton i*j = k", [0.0, 0.0, 0.0, 1.0], hamilton_product(i, j), results)

    # The raw pure-quaternion product carries the negative dot; the descriptor does not.
    _check(
        "raw pure product [x,x] = [-1,0,0,0]",
        [-1.0, 0, 0, 0],
        pure_quaternion_product(x, x),
        results,
    )
    _check("descriptor 90 deg rotation", [0.0, 0, 0, 1.0], relation_descriptor(x, y), results)
    _check("descriptor reversal", [-1.0, 0, 0, 0], relation_descriptor(x, -x), results)

    # Constant direction: first order is the identity relation, second order vanishes.
    constant = x.expand(50, 3).clone()
    _check(
        "constant direction first order",
        torch.tensor([1.0, 0, 0, 0]).expand(50, 4),
        first_order(constant, 5),
        results,
    )
    _check(
        "constant direction second order", torch.zeros(50, 4), second_order(constant, 5), results
    )

    # Uniform circular rotation: first order is constant, second order vanishes.
    omega, lag = 0.05, 5
    uniform = _spiral(omega * torch.arange(200.0))
    q_uniform = first_order(uniform, lag)[:150]
    _check(
        "uniform rotation first order is constant",
        torch.tensor([math.cos(omega * lag), 0, 0, math.sin(omega * lag)]).expand(150, 4),
        q_uniform,
        results,
    )
    _check(
        "uniform rotation second order = 0",
        torch.zeros(150, 4),
        second_order(uniform, lag)[:150],
        results,
    )

    # Accelerating rotation: first order varies, second order is nonzero.
    t = torch.arange(200.0)
    accelerating = _spiral(2e-4 * t.square())
    q_accelerating = first_order(accelerating, lag)[:150]
    second = second_order(accelerating, lag)[lag:150]
    results.append(
        {
            "test": "accelerating rotation first order varies",
            "expected": "spread of scalar part > 1e-3",
            "actual": round((q_accelerating[:, 0].max() - q_accelerating[:, 0].min()).item(), 6),
            "max_abs_error": None,
            "passed": bool(q_accelerating[:, 0].max() - q_accelerating[:, 0].min() > 1e-3),
        }
    )
    results.append(
        {
            "test": "accelerating rotation second order is nonzero",
            "expected": "max |s| > 1e-4",
            "actual": round(second.abs().max().item(), 6),
            "max_abs_error": None,
            "passed": bool(second.abs().max() > 1e-4),
        }
    )

    # dot^2 + ||cross||^2 = 1 on random directions (方案 §4.3).
    random_u = unit_direction(torch.randn(4, 300, 3, generator=torch.manual_seed(0)))
    error = relation_norm_error(first_order(random_u, 7)).item()
    results.append(
        {
            "test": "|| q_first || = 1 on random directions",
            "expected": f"max deviation <= {TOLERANCE}",
            "actual": error,
            "max_abs_error": error,
            "passed": error <= TOLERANCE,
        }
    )

    report = {"all_passed": all(item["passed"] for item in results), "checks": results}
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['test']}", flush=True)
        print(f"       expected: {item['expected']}", flush=True)
        print(f"       actual:   {item['actual']}", flush=True)
        if item["max_abs_error"] is not None:
            print(f"       max abs error: {item['max_abs_error']:.3g}", flush=True)
    print(json.dumps({"all_passed": report["all_passed"]}), flush=True)
    return report


def check_geometry(config, records=256):
    """Numerical report of the descriptor norm on real cached PTB-XL waveforms."""
    from .data import PTBXLDataset, load_manifest
    from .geometry import VCGGeometry, kors_transform

    manifest = load_manifest(config["data"])
    dataset = PTBXLDataset(config["data"]["cache"], "train", records, 0)
    geometry = VCGGeometry("first", [10, 20, 40, 80], manifest["stats"]["sampling_rate"])
    ecg = torch.stack([dataset[i]["ecg"] for i in range(len(dataset))])
    vcg = kors_transform(ecg, geometry.kors)
    u = unit_direction(vcg.transpose(1, 2))
    magnitude = torch.linalg.vector_norm(vcg, dim=1)
    report = {
        "records": len(dataset),
        "vcg_magnitude_mv": {
            "p01": magnitude.quantile(0.01).item(),
            "p50": magnitude.median().item(),
            "p99": magnitude.quantile(0.99).item(),
        },
        "unit_direction_norm_error": (torch.linalg.vector_norm(u, dim=-1) - 1).abs().max().item(),
        "scales": {},
    }
    for lag_ms, lag in zip(geometry.scales_ms, geometry.lags):
        q = first_order(u, lag)
        report["scales"][f"{lag_ms}ms"] = {
            "lag_samples": lag,
            "norm_max_abs_error": relation_norm_error(q).item(),
            "dot_mean": q[..., 0].mean().item(),
            "dot_p01": q[..., 0].quantile(0.01).item(),
            "cross_norm_mean": torch.linalg.vector_norm(q[..., 1:], dim=-1).mean().item(),
        }
    return report
