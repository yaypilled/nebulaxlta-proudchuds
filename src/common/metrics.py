"""Shared metric implementations.

Every subsystem and the tester score against the code in this file. A metric
computed two different ways in two places is a defect even when both are
correct, so nothing anywhere else may hand-roll or inline a metric.

Scope this session is rail only, so macro F1 is the only metric here. Door,
ACV and SHM metrics are deferred, not abandoned.

Reference for the rail metric --- Rail_Corrugation_Info_Kit.md section 4,
lines 127-129, quoted verbatim:

    Rail corrugation is graded on **macro F1** across the three classes
    (Normal, Side I, Side II) - not plain accuracy. Macro F1 computes the F1
    score for each class independently, then averages the three unweighted
    (each class counts equally regardless of how many examples it has).

Two consequences of that wording are load-bearing and are implemented
explicitly below:

* "each class counts equally regardless of how many examples it has" means the
  average is over the three CLASSES, not over the samples. A class absent from
  a particular fold still occupies a slot in the denominator.
* "a class the model never gets right (0 recall, 0 precision) contributes an
  F1 of 0 to the average" (line 134-135) fixes the degenerate case: an
  undefined 0/0 F1 is scored as 0.0, not dropped and not treated as 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score

__all__ = ["RAIL_CLASSES", "per_class_f1", "macro_f1", "macro_f1_from_per_class"]

#: The three rail classes, in the order the Info Kit lists them (section 4,
#: line 127). Fixing the order makes per-class output positionally stable and
#: guarantees every class occupies a slot in the macro average even when it is
#: absent from a given fold.
RAIL_CLASSES: tuple[str, ...] = ("Normal", "Side I", "Side II")


def per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = RAIL_CLASSES,
) -> dict[str, float]:
    """F1 for each class independently, keyed by class label.

    A class with no true samples and no predicted samples scores 0.0, per the
    Info Kit's rule that a never-correct class contributes 0 to the average.
    """
    scores = f1_score(
        y_true,
        y_pred,
        labels=list(labels),
        average=None,
        zero_division=0.0,
    )
    return {label: float(score) for label, score in zip(labels, scores)}


def macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = RAIL_CLASSES,
) -> float:
    """Macro F1: per-class F1 averaged unweighted over ``labels``.

    This is the rail ``primary_metric`` (Info Kit section 4, line 152).

    ``labels`` is passed explicitly so the denominator is always the full class
    count. Letting it default to the classes present in the data would shrink
    the denominator on a fold missing a minority class and silently inflate the
    score -- with 14 Side I samples across 272 files, that is a realistic fold.
    """
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(labels),
            average="macro",
            zero_division=0.0,
        )
    )


def macro_f1_from_per_class(per_class_scores: Sequence[float]) -> float:
    """Macro-average a set of already-computed per-class F1 scores.

    Exists because the Info Kit's worked example (section 4, lines 137-145) is
    stated as three per-class F1 values rather than as predictions, so this is
    the exact operation that example pins down. Production code paths should
    call :func:`macro_f1` on labels instead.
    """
    scores = np.asarray(per_class_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("per_class_scores must not be empty")
    return float(scores.mean())
