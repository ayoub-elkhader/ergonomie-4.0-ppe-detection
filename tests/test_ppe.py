"""Decision tests for ppe.check_ppe (no model needed)."""
import pytest

import ppe
from ppe import Detection as D

ALL_WORN = [D("avec_casque", .9), D("avec_gants", .8), D("tablier_ferme", .8),
            D("chaussure_adequat", .7)]


def status(report, label):
    return next(i["status"] for i in report["items"] if i["label"] == label)


def test_helmet_only_is_incomplete_not_compliant():
    r = ppe.check_ppe([D("avec_casque", .95)], n_persons=1, decision_conf=.5)
    assert r["overall"] == ppe.INCOMPLETE
    assert status(r, "Welding helmet") == ppe.WORN
    for label in ("Gloves", "Apron", "Safety shoes"):
        assert status(r, label) == ppe.NOT_OBSERVED


def test_no_detection_at_all_is_incomplete():
    r = ppe.check_ppe([], n_persons=1, decision_conf=.5)
    assert r["overall"] == ppe.INCOMPLETE
    assert all(i["status"] == ppe.NOT_OBSERVED for i in r["items"])


def test_all_items_confirmed_one_person_is_no_anomaly():
    r = ppe.check_ppe(ALL_WORN, n_persons=1, decision_conf=.5)
    assert r["overall"] == ppe.NO_ANOMALY


def test_confirmed_negative_is_anomaly_even_if_other_items_missing():
    r = ppe.check_ppe([D("sans_gants", .7)], n_persons=1, decision_conf=.5)
    assert r["overall"] == ppe.ANOMALY
    assert status(r, "Gloves") == ppe.MISSING


@pytest.mark.parametrize("cls", ["sans_tablier", "tablier_ouvert"])
def test_apron_violations(cls):
    r = ppe.check_ppe(ALL_WORN[:2] + ALL_WORN[3:] + [D(cls, .6)], n_persons=1, decision_conf=.5)
    assert r["overall"] == ppe.ANOMALY and status(r, "Apron") == ppe.MISSING


def test_confirmed_hazard_is_anomaly():
    r = ppe.check_ppe(ALL_WORN + [D("presence_cheveux", .6)], n_persons=1, decision_conf=.5)
    assert r["overall"] == ppe.ANOMALY and r["hazards"]


def test_uncertain_hazard_makes_evaluation_incomplete():
    r = ppe.check_ppe(ALL_WORN + [D("presence_accessoires", .3)], 1, .5, .25)
    assert r["overall"] == ppe.INCOMPLETE and r["uncertain_hazards"]


def test_low_score_positive_is_uncertain_not_worn():
    r = ppe.check_ppe([D("avec_casque", .4)] + ALL_WORN[1:], 1, .5, .25)
    assert status(r, "Welding helmet") == ppe.UNCERTAIN
    assert r["overall"] == ppe.INCOMPLETE


def test_weak_negative_downgrades_confirmed_positive_to_uncertain():
    r = ppe.check_ppe(ALL_WORN + [D("sans_casque", .3)], 1, .5, .25)
    assert status(r, "Welding helmet") == ppe.UNCERTAIN
    assert r["overall"] == ppe.INCOMPLETE


def test_conflicting_confirmed_detections_are_uncertain():
    r = ppe.check_ppe(ALL_WORN[1:] + [D("avec_casque", .8), D("sans_casque", .7)], 1, .5)
    assert status(r, "Welding helmet") == ppe.UNCERTAIN


def test_scores_below_uncertain_threshold_are_ignored():
    r = ppe.check_ppe(ALL_WORN + [D("sans_gants", .1)], 1, .5, .25)
    assert r["overall"] == ppe.NO_ANOMALY


@pytest.mark.parametrize("n", [0, 2, 3, None])
def test_not_exactly_one_person_is_incomplete(n):
    r = ppe.check_ppe(ALL_WORN, n_persons=n, decision_conf=.5)
    assert r["overall"] == ppe.INCOMPLETE and r["scope_issues"]


def test_two_helmets_flag_several_people_even_if_counter_says_one():
    r = ppe.check_ppe(ALL_WORN + [D("avec_casque", .8)], n_persons=1, decision_conf=.5)
    assert r["overall"] == ppe.INCOMPLETE


def test_anomaly_with_several_people_stays_anomaly_with_scope_note():
    r = ppe.check_ppe([D("sans_casque", .9)], n_persons=2, decision_conf=.5)
    assert r["overall"] == ppe.ANOMALY and r["scope_issues"]


def test_decision_threshold_changes_outcome():
    dets = [D("avec_casque", .55)] + ALL_WORN[1:]
    assert ppe.check_ppe(dets, 1, .5)["overall"] == ppe.NO_ANOMALY
    assert ppe.check_ppe(dets, 1, .6)["overall"] == ppe.INCOMPLETE
