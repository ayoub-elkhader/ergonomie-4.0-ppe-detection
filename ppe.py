"""PPE decision logic, independent of Streamlit and of the detector.

Inputs are plain detections (class name + detector score) and the number of
people found in the image. The output is a checklist with an overall status:

    ANOMALY        at least one confirmed "missing / badly worn" detection or
                   confirmed hazard (hair, accessories);
    NO_ANOMALY     every required item is confirmed as worn and nothing
                   negative or uncertain was found, for exactly one operator.
                   This means "no anomaly detected among the observed items",
                   not a certification of compliance;
    INCOMPLETE     anything else: an item is not observed or only uncertain,
                   detections conflict, or the image does not contain exactly
                   one operator.

Detector scores are thresholds for evidence, never probabilities of compliance.
"""
from dataclasses import dataclass

import settings

ANOMALY, NO_ANOMALY, INCOMPLETE = "anomaly", "no_anomaly", "incomplete"

# Item statuses
WORN = "worn"                 # confirmed positive evidence, nothing against it
MISSING = "missing"           # confirmed negative evidence
UNCERTAIN = "uncertain"       # only low-score evidence, or conflicting evidence
NOT_OBSERVED = "not_observed"  # no evidence at all (out of view, hidden, missed)


@dataclass(frozen=True)
class Detection:
    name: str
    score: float


def _evidence(detections, classes, decision_conf, uncertain_conf):
    """(best confirmed score, best uncertain score) for the given classes."""
    confirmed = [d.score for d in detections if d.name in classes and d.score >= decision_conf]
    uncertain = [d.score for d in detections
                 if d.name in classes and uncertain_conf <= d.score < decision_conf]
    return (max(confirmed) if confirmed else None, max(uncertain) if uncertain else None)


def check_item(detections, ok_classes, bad_classes, decision_conf, uncertain_conf):
    """Status and explanation for one required item."""
    ok_c, ok_u = _evidence(detections, ok_classes, decision_conf, uncertain_conf)
    bad_c, bad_u = _evidence(detections, bad_classes, decision_conf, uncertain_conf)
    bad_names = sorted({d.name for d in detections
                        if d.name in bad_classes and d.score >= decision_conf})

    if bad_c is not None and ok_c is not None:
        return UNCERTAIN, "conflicting detections (worn and missing)"
    if bad_c is not None:
        return MISSING, ", ".join(settings.VIOLATION_LABELS.get(n, n) for n in bad_names)
    if ok_c is not None and bad_u is None:
        return WORN, f"detected (score {ok_c:.2f})"
    if ok_c is not None:
        return UNCERTAIN, f"detected, but a weak 'missing' detection exists (score {bad_u:.2f})"
    if ok_u is not None or bad_u is not None:
        best = max(s for s in (ok_u, bad_u) if s is not None)
        return UNCERTAIN, f"only low-score detections (best {best:.2f})"
    return NOT_OBSERVED, "not observed (out of view, hidden or missed)"


def check_ppe(detections, n_persons, decision_conf=None, uncertain_conf=None):
    """Build the PPE report for one image.

    detections: iterable of Detection (all detections above uncertain_conf).
    n_persons:  number of people found by the person detector, or None if unknown.
    """
    decision_conf = settings.DEFAULT_DECISION_CONFIDENCE if decision_conf is None else decision_conf
    uncertain_conf = settings.UNCERTAIN_CONFIDENCE if uncertain_conf is None else uncertain_conf
    uncertain_conf = min(uncertain_conf, decision_conf)
    detections = list(detections)

    items = []
    for label, ok_classes, bad_classes in settings.PPE_CHECKLIST:
        status, detail = check_item(detections, ok_classes, bad_classes,
                                    decision_conf, uncertain_conf)
        items.append({"label": label, "status": status, "detail": detail})

    hazards, uncertain_hazards = [], []
    for cls, message in settings.HAZARD_CLASSES.items():
        conf, unc = _evidence(detections, [cls], decision_conf, uncertain_conf)
        if conf is not None:
            hazards.append(message)
        elif unc is not None:
            uncertain_hazards.append(message)

    heads = sum(1 for d in detections
                if d.name in settings.HEAD_CLASSES and d.score >= decision_conf)
    scope = []
    if n_persons is None:
        scope.append("number of people unknown")
    elif n_persons == 0:
        scope.append("no person detected")
    elif n_persons > 1:
        scope.append(f"{n_persons} people detected: equipment cannot be attributed to one operator")
    if heads > 1 and (n_persons or 0) <= 1:
        scope.append(f"{heads} head-level detections suggest several people")

    if any(i["status"] == MISSING for i in items) or hazards:
        overall = ANOMALY
    elif scope or uncertain_hazards or any(i["status"] != WORN for i in items):
        overall = INCOMPLETE
    else:
        overall = NO_ANOMALY

    return {
        "overall": overall,
        "items": items,
        "hazards": hazards,
        "uncertain_hazards": uncertain_hazards,
        "scope_issues": scope,
        "n_persons": n_persons,
        "decision_conf": decision_conf,
        "uncertain_conf": uncertain_conf,
    }


def detections_from_result(result, min_score):
    """Convert an Ultralytics result into Detection objects above min_score."""
    names = result.names
    return [Detection(names[int(c)], float(s))
            for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
            if s >= min_score]


def count_persons(result, class_id=None, min_score=None):
    """Number of 'person' boxes in an Ultralytics COCO result."""
    class_id = settings.PERSON_CLASS_ID if class_id is None else class_id
    min_score = settings.PERSON_CONFIDENCE if min_score is None else min_score
    return sum(1 for c, s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
               if int(c) == class_id and s >= min_score)
