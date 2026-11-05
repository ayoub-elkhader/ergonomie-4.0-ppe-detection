"""Geometric tests of the posture indicators on synthetic skeletons.

Coordinates use the image convention: x to the right, y DOWN. The same
points are used as 2D pixels and 3D world landmarks (z = 0).
"""
import math

import numpy as np
import pytest

import posture
from posture import (ELEVATED, HIGH, LOW, UNAVAILABLE, angle_between, compute_indicators,
                     joint_flexion_3d, trunk_inclination_2d)


def rot(v, deg):
    """Rotate a 2D vector (image coords, y down) by deg towards +x when pointing up."""
    a = math.radians(deg)
    x, y = v
    return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))


def skeleton(shoulder, hip, knee, ankle, ear, half_width=10.0, right_offset=None):
    """33x2 / 33x3 arrays with left/right points side by side."""
    pts = np.zeros((33, 2))
    right_offset = right_offset or {}
    for name, (l_idx, r_idx), p in (("ear", posture.EAR, ear), ("shoulder", posture.SHOULDER, shoulder),
                                     ("hip", posture.HIP, hip), ("knee", posture.KNEE, knee),
                                     ("ankle", posture.ANKLE, ankle)):
        pts[l_idx] = (p[0] - half_width, p[1])
        rp = right_offset.get(name, p)
        pts[r_idx] = (rp[0] + half_width, rp[1])
    pts3 = np.hstack([pts, np.zeros((33, 1))])
    return pts, pts3, np.ones(33)


STANDING = dict(ear=(0, -170), shoulder=(0, -150), hip=(0, 0), knee=(0, 120), ankle=(0, 240))
SEATED_UPRIGHT = dict(ear=(0, -170), shoulder=(0, -150), hip=(0, 0), knee=(120, 0), ankle=(120, 120))


def inclined_whole_body(deg):
    p = {k: rot(v, deg) for k, v in STANDING.items()}
    return p


def ind(res, name):
    return res["indicators"][name]


def test_standing_upright():
    r = compute_indicators(*skeleton(**STANDING))
    for name in posture.INDICATORS:
        assert ind(r, name)["value"] == pytest.approx(0, abs=1e-4)
    assert r["overall"] == LOW


def test_seated_upright_trunk_vertical_hip_90():
    r = compute_indicators(*skeleton(**SEATED_UPRIGHT))
    assert ind(r, "trunk_inclination")["value"] == pytest.approx(0, abs=1e-4)
    assert ind(r, "hip_flexion")["value"] == pytest.approx(90, abs=1e-4)
    assert ind(r, "knee_flexion")["value"] == pytest.approx(90, abs=1e-4)
    assert r["overall"] == LOW  # sitting upright is not flagged


def test_whole_body_inclined_45_trunk_is_45_hip_is_0():
    r = compute_indicators(*skeleton(**inclined_whole_body(45)))
    assert ind(r, "trunk_inclination")["value"] == pytest.approx(45, abs=1e-4)
    assert ind(r, "hip_flexion")["value"] == pytest.approx(0, abs=1e-4)
    assert ind(r, "trunk_inclination")["level"] == ELEVATED


def test_bent_forward_70_standing_legs_is_high():
    p = dict(STANDING)
    p["shoulder"] = rot((0, -150), 70)
    p["ear"] = rot((0, -170), 70)
    r = compute_indicators(*skeleton(**p))
    assert ind(r, "trunk_inclination")["value"] == pytest.approx(70, abs=1e-4)
    assert ind(r, "hip_flexion")["value"] == pytest.approx(70, abs=1e-4)
    assert r["overall"] == HIGH


def test_neck_flexion_relative_to_trunk():
    p = dict(STANDING)
    p["ear"] = np.add((0, -150), rot((0, -20), 30))  # head 30 deg forward, trunk vertical
    r = compute_indicators(*skeleton(**p))
    assert ind(r, "neck_flexion")["value"] == pytest.approx(30, abs=1e-4)
    assert ind(r, "neck_flexion")["level"] == HIGH


def test_asymmetry_reported_and_worst_side_used():
    pts, pts3, vis = skeleton(**STANDING, right_offset={"knee": (60, 100)})
    r = compute_indicators(pts, pts3, vis)
    k = ind(r, "knee_flexion")
    assert k["asymmetry"] and k["value"] == max(k["left"], k["right"]) and k["left"] == pytest.approx(0)


def test_low_visibility_side_is_ignored_then_unavailable():
    pts, pts3, vis = skeleton(**inclined_whole_body(30))
    vis[posture.SHOULDER[0]] = 0.1           # left shoulder hidden
    r = compute_indicators(pts, pts3, vis)
    t = ind(r, "trunk_inclination")
    assert t["left"] is None and t["right"] == pytest.approx(30) and t["value"] == pytest.approx(30)
    vis[posture.SHOULDER[1]] = 0.1           # both hidden
    r = compute_indicators(pts, pts3, vis)
    assert ind(r, "trunk_inclination")["value"] is None
    assert ind(r, "trunk_inclination")["level"] == UNAVAILABLE


def test_all_points_hidden_overall_unavailable():
    pts, pts3, vis = skeleton(**STANDING)
    r = compute_indicators(pts, pts3, np.zeros(33))
    assert r["overall"] == UNAVAILABLE
    assert all(i["value"] is None for i in r["indicators"].values())


def test_degenerate_coincident_points_give_none():
    assert angle_between((0, 0), (1, 0)) is None
    assert trunk_inclination_2d((5, 5), (5, 5)) is None
    assert joint_flexion_3d((0, 0, 0), (0, 0, 0), (1, 0, 0)) is None
    pts, pts3, vis = skeleton(**STANDING)
    pts[:] = 0
    pts3[:] = 0
    r = compute_indicators(pts, pts3, vis)
    assert r["overall"] == UNAVAILABLE


def test_nan_coordinates_do_not_crash():
    pts, pts3, vis = skeleton(**STANDING)
    pts3[posture.KNEE[0]] = np.nan
    r = compute_indicators(pts, pts3, vis)
    assert ind(r, "knee_flexion")["right"] == pytest.approx(0)


def test_thresholds_are_configurable():
    r = compute_indicators(*skeleton(**inclined_whole_body(45)),
                           thresholds={"trunk_inclination": (50, 70), "neck_flexion": (10, 20),
                                       "hip_flexion": (45, 90), "knee_flexion": (30, 60)})
    assert ind(r, "trunk_inclination")["level"] == LOW


def test_exposure_uses_timestamps_and_skips_gaps():
    e = posture.ExposureTracker(max_gap=1.0)
    elevated = compute_indicators(*skeleton(**inclined_whole_body(45)))
    for t in (0.0, 0.5, 1.0, 5.0, 5.5):      # 1.0 -> 5.0 is a gap, not counted
        e.update(elevated, t)
    assert e.total == pytest.approx(1.5)
    assert e.summary()["trunk_inclination"]["elevated"] == pytest.approx(1.5)
    e.update(None, 6.0)
    assert e.summary()["trunk_inclination"]["unavailable"] == pytest.approx(0.5)
