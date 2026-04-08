"""Optional matplotlib export for plot actions (declarations → real PNG files).

The environment normally only *records* plot intent for grading. When
``AUTODATALAB_PLOT_DIR`` is set, the server can call :func:`save_plot_to_png`
after a ``plot`` action.

Install: ``pip install -e ".[plot]"`` (adds matplotlib).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

PlotKind = Literal["scatter", "bar", "histogram"]


def _point_label_column(df: pd.DataFrame, x: str, y: str) -> Optional[str]:
    """Prefer a label column for annotating scatter points when not used as an axis."""
    for c in ("Name", "name", "Product", "product"):
        if c in df.columns and c not in (x, y):
            return c
    return None


def _series_numeric_or_datetime(s: pd.Series) -> pd.Series:
    """Use numeric values when possible; otherwise parse datetimes (e.g. ``OrderDate`` strings)."""
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return num
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if dt.notna().any():
        return dt
    return num


def save_plot_to_png(
    df: pd.DataFrame,
    plot_type: Optional[str],
    x: Optional[str],
    y: Optional[str],
    out_path: Path,
    *,
    title: str = "",
) -> None:
    """Render a simple figure from the current table and write *out_path* (``.png``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pt = (plot_type or "scatter").lower()
    w, h = (8.5, 5.2) if pt == "scatter" else (7.6, 4.8)
    fig, ax = plt.subplots(figsize=(w, h))
    title = title or f"{pt}: {x!r} vs {y!r}"
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    if pt == "scatter":
        if not x or not y or x not in df.columns or y not in df.columns:
            raise ValueError(f"scatter requires valid x,y columns; got x={x!r} y={y!r}")
        xs = _series_numeric_or_datetime(df[x])
        ys = pd.to_numeric(df[y], errors="coerce")
        view = pd.DataFrame({"_x": xs, "_y": ys}).dropna().copy()
        if pd.api.types.is_datetime64_any_dtype(view["_x"]):
            view = view.groupby("_x", as_index=False)["_y"].sum().sort_values("_x")
            ax.plot(view["_x"], view["_y"], color="#93c5fd", linewidth=1.4, zorder=1)
        ax.scatter(view["_x"], view["_y"], s=42, color="#2563eb", edgecolors="white", linewidths=0.6, alpha=0.92, zorder=2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        label_col = _point_label_column(df, x, y)
        if label_col is not None:
            for i in range(len(df)):
                lab = df[label_col].iloc[i]
                if pd.isna(lab) or (pd.isna(xs.iloc[i]) and pd.isna(ys.iloc[i])):
                    continue
                if pd.isna(xs.iloc[i]) or pd.isna(ys.iloc[i]):
                    continue
                ax.annotate(
                    str(lab),
                    (xs.iloc[i], float(ys.iloc[i])),
                    fontsize=7,
                    alpha=0.78,
                    xytext=(4, 4),
                    textcoords="offset points",
                    zorder=3,
                )
            ax.set_title(f"{title} (labels: {label_col})")
        else:
            ax.set_title(title)
    elif pt == "bar":
        if not x or x not in df.columns:
            raise ValueError(f"bar requires valid column x={x!r}")
        if y and y in df.columns:
            # Category vs sales / revenue: aggregate numeric y per category on x
            vals = pd.to_numeric(df[y], errors="coerce")
            g = df.assign(_y=vals).groupby(x, dropna=False, sort=True)["_y"].sum()
            g = g.dropna(how="all")
            g = g.sort_values(ascending=False).head(20)
            g.plot(kind="bar", ax=ax, color="#2563eb", edgecolor="#1e3a8a", width=0.72)
            ax.set_ylabel(y)
        else:
            s = df[x].value_counts().head(20)
            s.plot(kind="bar", ax=ax, color="#2563eb", edgecolor="#1e3a8a", width=0.72)
        ax.set_xlabel(x)
        ax.tick_params(axis="x", rotation=25)
    elif pt == "histogram":
        col = x or y
        if not col or col not in df.columns:
            raise ValueError(f"histogram requires a column; got x={x!r} y={y!r}")
        ax.hist(
            pd.to_numeric(df[col], errors="coerce").dropna(),
            bins=min(20, max(5, len(df))),
            color="#2563eb",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xlabel(col)
    else:
        raise ValueError(f"unsupported plot_type: {plot_type!r}")

    if pt != "scatter":
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: render a CSV + plot spec to PNG (for agent pipelines / debugging)."""
    p = argparse.ArgumentParser(
        description="Render a plot from a CSV file (optional artifact export for AutoDataLab)."
    )
    p.add_argument("csv", type=Path, help="Path to CSV (same shape as env working table)")
    p.add_argument("plot_type", choices=("scatter", "bar", "histogram"))
    p.add_argument("x", help="X column (or primary column for histogram)")
    p.add_argument("y", nargs="?", default=None, help="Y column (scatter only)")
    p.add_argument("-o", "--output", type=Path, default=Path("plot_out.png"))
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    try:
        save_plot_to_png(df, args.plot_type, args.x, args.y, args.output)
    except ImportError:
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


def _entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
