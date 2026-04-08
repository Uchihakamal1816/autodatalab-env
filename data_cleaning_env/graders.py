"""Deterministic graders for each task (scores in [0.0, 1.0])."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _prepare_columns(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    grade_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select required grading columns.

    Unlike the older shared-column behavior, this returns an empty result when any required
    ground-truth column is missing from the prediction. That closes a loophole where dropping
    hard columns could still yield a misleadingly high score on the remaining subset.
    """
    cols = list(grade_columns) if grade_columns else list(gt.columns)
    if not cols:
        return pred.iloc[:0], gt.iloc[:0]
    if any(c not in gt.columns for c in cols):
        return pred.iloc[:0], gt.iloc[:0]
    if any(c not in pred.columns for c in cols):
        return pred.iloc[:0], gt.iloc[:0]
    return pred[cols].copy(), gt[cols].copy()


def _sort_canonically(df: pd.DataFrame) -> pd.DataFrame:
    """Make comparison row-order invariant without mutating the caller's table."""
    if df.empty or len(df.columns) == 0:
        return df.reset_index(drop=True)
    keys = pd.DataFrame(
        {c: df[c].map(lambda v: "" if pd.isna(v) else str(v)) for c in df.columns},
        index=df.index,
    )
    order = keys.sort_values(by=list(keys.columns), kind="mergesort").index
    return df.loc[order].reset_index(drop=True)


def cleaning_match_score(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    grade_columns: Optional[Sequence[str]] = None,
    sort_rows: bool = False,
) -> float:
    """Fraction of cells equal to ground truth (after column alignment and row count match).

    Numeric columns use a relative tolerance of 1e-6 to tolerate floating-point imprecision
    from mean imputation (mean value can differ by a ULP between a freshly computed run and
    a value round-tripped through CSV).
    """
    import numpy as np

    p, g = _prepare_columns(pred, gt, grade_columns=grade_columns)
    if sort_rows:
        p = _sort_canonically(p)
        g = _sort_canonically(g)
    if p.shape != g.shape or p.size == 0:
        return 0.0

    total = p.size
    matched = 0
    for col in p.columns:
        ps = p[col]
        gs = g[col]
        # Try numeric comparison with tolerance for float-imputed columns
        pn = pd.to_numeric(ps, errors="coerce")
        gn = pd.to_numeric(gs, errors="coerce")
        both_numeric = pn.notna() & gn.notna()
        both_nan = ps.isna() & gs.isna()
        if both_numeric.any():
            # Numeric match: relative tol 1e-6, absolute tol 1e-3 (handles near-zero)
            num_close = np.isclose(pn.values, gn.values, rtol=1e-6, atol=1e-3, equal_nan=False)
            # Non-numeric (e.g. string) cells: exact string match
            str_match = ps.astype(str).values == gs.astype(str).values
            cell_match = (both_numeric.values & num_close) | (both_nan.values) | (
                ~both_numeric.values & ~both_nan.values & str_match
            )
        else:
            p2 = ps.astype(object).where(pd.notna(ps), None)
            g2 = gs.astype(object).where(pd.notna(gs), None)
            cell_match = (p2.values == g2.values) | (pd.isna(p2.values) & pd.isna(g2.values))
        matched += int(cell_match.sum())
    return matched / total


def plot_match_score(
    plot_type: str | None,
    x: str | None,
    y: str | None,
    expected: List[Dict[str, Any]],
) -> float:
    """1.0 if the chosen plot matches any expected spec; partial credit otherwise."""
    if not expected:
        return 1.0
    best = 0.0
    for exp in expected:
        score = 0.0
        if plot_type == exp.get("type"):
            score += 0.45
        if x == exp.get("x") and y == exp.get("y"):
            score += 0.45
        if plot_type == exp.get("type") and x == exp.get("x"):
            score += 0.1
        best = max(best, min(score, 1.0))
    return best


def plot_coverage_score(
    plot_history: List[Dict[str, Any]],
    expected: List[Dict[str, Any]],
) -> float:
    """Fraction of expected plot specs that are matched by at least one declared plot in history."""
    if not expected:
        return 1.0
    if not plot_history:
        return 0.0
    matched = 0
    for exp in expected:
        for h in plot_history:
            pt = h.get("plot_type")
            x = h.get("x")
            y = h.get("y")
            if plot_match_score(pt, x, y, [exp]) >= 0.85:
                matched += 1
                break
    return matched / len(expected)


def grade_task(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    metadata: Dict[str, Any],
    plot_action: Dict[str, Any] | None,
    plot_history: Optional[List[Dict[str, Any]]] = None,
) -> Optional[float]:
    """
    Combined score in [0, 1]: cleaning table vs GT + optional plot spec.

    Returns ``None`` when there is no ground-truth table (empty or missing file), so episodes
    still run but are not numerically graded.

    For tasks without expected plots, plot contribution is skipped (reweighted to cleaning only).
    """
    if gt is None or gt.empty:
        return None
    grade_columns = metadata.get("grade_columns")
    sort_rows = bool(metadata.get("sort_rows", False))
    clean = cleaning_match_score(
        pred,
        gt,
        grade_columns=grade_columns,
        sort_rows=sort_rows,
    )
    expected = metadata.get("expected_plots") or []
    if not expected:
        return clean
    plot_weight = float(metadata.get("plot_weight", 0.2))
    plot_weight = min(max(plot_weight, 0.0), 1.0)
    clean_weight = 1.0 - plot_weight
    hist = plot_history or []
    if len(expected) > 1:
        pm = plot_coverage_score(hist, expected)
        if pm == 0.0 and plot_action is not None:
            pm = plot_match_score(
                plot_action.get("plot_type"),
                plot_action.get("x"),
                plot_action.get("y"),
                expected,
            )
        return clean_weight * clean + plot_weight * pm
    if plot_action is None and not hist:
        return clean_weight * clean
    pm = plot_match_score(
        plot_action.get("plot_type") if plot_action else None,
        plot_action.get("x") if plot_action else None,
        plot_action.get("y") if plot_action else None,
        expected,
    )
    if hist:
        pm = max(pm, plot_coverage_score(hist, expected))
    return clean_weight * clean + plot_weight * pm
