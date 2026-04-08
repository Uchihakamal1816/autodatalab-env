"""
AutoDataLab — Gradio Web UI
E-commerce Data Analyst · RL Environment Demo

Modes:
  🤖 Oracle  — deterministic expert policy (no API key needed)
  🧠 LLM     — Groq drives the agent (paste GROQ_API_KEY)

Local:
  python app.py            # http://127.0.0.1:7861
  python app.py --share    # public Gradio link

HF Spaces: set secret GROQ_API_KEY in Space settings.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_cleaning_env.models import DataCleaningAction
from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment

# ── constants ────────────────────────────────────────────────────────────────
TASKS = ["easy", "medium", "medium_plus", "hard", "expert"]

TASK_DESCRIPTIONS = {
    "easy":        "🟢 **Task 1 — Data Cleaning** — dedupe · impute Price · Expiry only warns",
    "medium":      "🟡 **Task 2 — Business Metrics** — compute category-wise revenue",
    "medium_plus": "🟠 **Task 2+ — Full KPIs** — TotalRevenue · AvgOrderValue",
    "hard":        "🔴 **Task 3 — Insight + Viz** — clean · derive Revenue · 2 charts",
    "expert":      "⚫ **Task 4 — Expert Pipeline** — full clean + revenue + 2 charts",
}

PIPELINE_STEPS = {
    "easy":        ["remove_duplicates", "fill_missing (Price, mean)", "submit"],
    "medium":      ["compute_metrics", "submit"],
    "medium_plus": ["compute_kpis", "submit"],
    "hard":        ["remove_duplicates", "fill_missing (Price, mean)", "derive_revenue", "plot scatter — OrderDate × Revenue", "plot bar — Category × Revenue", "submit"],
    "expert":      ["remove_duplicates", "fill_missing (Price, mean)", "derive_revenue", "plot scatter — OrderDate × Revenue", "plot bar — Category × Revenue", "submit"],
}

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ── dark-mode CSS ─────────────────────────────────────────────────────────────
CSS = """
footer { display:none !important; }
.log-box { font-size:13px; }
/* Pipeline hint — dark box */
.hint-box {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
    margin-top: 6px;
}
.hint-box p, .hint-box li, .hint-box code {
    color: #e2e8f0 !important;
}
.hint-box code {
    background: #1e293b !important;
    border: 1px solid #475569 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
}
/* Plot gallery items */
.plot-gallery img { border-radius: 8px; border: 1px solid #334155; }
"""

# ── helpers ───────────────────────────────────────────────────────────────────
def _df_to_html(df: pd.DataFrame, max_rows: int = 15) -> str:
    display = df.head(max_rows).copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].round(2)
    display = display.fillna("—")

    CELL_BORDER = "1px solid #2d5080"
    TEXT_COLOR  = "#e2e8f0"
    HEADER_BG   = "#2563eb"

    rows_html = ""
    for i, (_, row) in enumerate(display.iterrows()):
        bg = "#162c47" if i % 2 == 0 else "#1e3a5f"
        cells = "".join(
            f"<td style='padding:7px 14px;border-bottom:{CELL_BORDER};"
            f"font-size:13px;white-space:nowrap;color:{TEXT_COLOR};background:{bg};'>{v}</td>"
            for v in row
        )
        rows_html += f"<tr>{cells}</tr>"

    header = "".join(
        f"<th style='padding:8px 14px;background:{HEADER_BG};color:#ffffff;"
        f"font-weight:700;font-size:13px;white-space:nowrap;text-align:left;"
        f"border-bottom:2px solid #1d4ed8;position:sticky;top:0;z-index:2;'>{c}</th>"
        for c in display.columns
    )
    extra = ""
    if len(df) > max_rows:
        extra = (
            f"<p style='color:#94a3b8;font-size:12px;margin:6px 0 0 4px;'>"
            f"Showing first {max_rows} of <b>{len(df)}</b> rows</p>"
        )
    return (
        f"<div style='overflow:auto;max-height:420px;border-radius:8px;border:1px solid #2d5080;'>"
        f"<table style='border-collapse:separate;border-spacing:0;width:100%;min-width:780px;'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>{extra}"
    )


def _issues_html(issues: List[str]) -> str:
    if not issues:
        return (
            "<div style='background:#052e16;border:1px solid #166534;border-radius:8px;padding:12px 16px;'>"
            "<span style='color:#4ade80;font-weight:600;font-size:14px;'>✅ No issues detected — table looks clean</span>"
            "</div>"
        )
    items = "".join(
        f"<li style='margin:5px 0;font-size:13px;color:#fca5a5;'>{i}</li>" for i in issues
    )
    return (
        f"<div style='background:#1c0a0a;border:1px solid #7f1d1d;border-radius:8px;padding:12px 16px;'>"
        f"<b style='color:#f87171;'>⚠️ Issues detected:</b>"
        f"<ul style='margin:8px 0 0 0;padding-left:20px;'>{items}</ul></div>"
    )


def _score_html(score: Optional[float], done: bool) -> str:
    if not done:
        return ""
    if score is None:
        return "<div style='color:#64748b;padding:10px;'>Episode ended — no ground truth file.</div>"
    pct = score * 100
    color  = "#16a34a" if pct >= 90 else "#d97706" if pct >= 60 else "#dc2626"
    emoji  = "🏆" if pct >= 90 else "📊" if pct >= 60 else "📉"
    return (
        f"<div style='background:{color};color:white;border-radius:10px;"
        f"padding:14px 20px;font-size:24px;font-weight:700;text-align:center;'>"
        f"{emoji} Final Score: {pct:.1f}%</div>"
    )


def _pipeline_html(task: str) -> str:
    """Dark-mode styled numbered pipeline list."""
    steps = PIPELINE_STEPS[task]
    items = "".join(
        f"<li style='margin:6px 0;color:#e2e8f0;font-size:14px;'>"
        f"<code style='background:#1e293b;border:1px solid #475569;padding:2px 8px;"
        f"border-radius:4px;color:#93c5fd;font-size:13px;'>{s}</code></li>"
        for s in steps
    )
    return (
        f"<div style='background:#0f172a;border:1px solid #334155;border-radius:10px;"
        f"padding:14px 18px;margin-top:6px;'>"
        f"<p style='color:#94a3b8;font-size:12px;margin:0 0 8px 0;text-transform:uppercase;"
        f"letter-spacing:.05em;'>Pipeline steps</p>"
        f"<ol style='margin:0;padding-left:20px;'>{items}</ol></div>"
    )


def _render_plot_fig(df: pd.DataFrame, plot_type: str, x_col: str, y_col: str):
    """Return a matplotlib figure rendered from the current df."""
    try:
        from data_cleaning_env.plot_artifacts import save_plot_to_png
        tmp = Path(tempfile.mktemp(suffix=".png"))
        save_plot_to_png(df, plot_type, x_col, y_col, tmp)
        img = plt.imread(str(tmp))
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0f172a")
        ax.imshow(img)
        ax.axis("off")
        plt.tight_layout(pad=0)
        tmp.unlink(missing_ok=True)
        return fig
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(6, 2), facecolor="#0f172a")
        ax.text(0.5, 0.5, f"Plot error:\n{exc}", ha="center", va="center",
                transform=ax.transAxes, color="white", fontsize=10)
        ax.axis("off")
        return fig


def _generate_report(sess: "Session") -> Optional[str]:
    """Build a Word docx directly from session state — no EpisodeTraceBuilder needed."""
    if sess.obs is None:
        return None
    try:
        from data_cleaning_env.episode_report import EpisodeTrace, write_episode_docx
    except ImportError:
        return None

    obs = sess.obs

    # Reconstruct operations list from the env history (JSON action strings)
    operations: List[str] = []
    for raw in (obs.history or []):
        try:
            d = json.loads(raw)
            from data_cleaning_env.episode_report import _brief_action_dict
            operations.append(_brief_action_dict(d))
        except Exception:
            operations.append(str(raw))

    # Get final table from env
    try:
        df_final = sess.env._df
        preview   = df_final.head(15).fillna("—").to_dict(orient="records")
        row_count = len(df_final)
        columns   = list(df_final.columns)
    except Exception:
        preview   = obs.preview or []
        row_count = len(preview)
        columns   = list(preview[0].keys()) if preview else []

    score = obs.terminal_grader_score if obs.done else None

    trace = EpisodeTrace(
        task=sess.task,
        mode="ui",
        model_name=None,
        instruction=obs.instruction or "",
        terminal_grader_score=score,
        operations_done=operations or ["(no operations recorded)"],
        final_row_count=row_count,
        final_columns=columns,
        final_preview=preview,
        remaining_issues=obs.issues or [],
        remaining_policy_warnings=getattr(obs, "policy_warnings", []) or [],
    )

    out = Path(tempfile.mktemp(suffix=".docx"))
    try:
        write_episode_docx(trace, out)
        return str(out)
    except Exception as e:
        print(f"[report] write failed: {e}")
        return None


# ── session state ──────────────────────────────────────────────────────────────
class Session:
    def __init__(self):
        self.env = DataCleaningEnvironment()
        self.task: str = "easy"
        self.obs: Any = None
        self.initial_obs: Any = None
        self.log: List[str] = []
        self.plot_figs: List[Any] = []       # matplotlib figures
        self.plot_paths: List[str] = []      # temp PNG paths for gallery
        self._trace_steps: List[tuple] = []  # for report

    def reset(self, task: str):
        self.env = DataCleaningEnvironment()
        self.task = task
        self.obs = self.env.reset(task=task)
        self.initial_obs = self.obs
        self.log = [f"🔄 **Reset** — task `{task}` · {self.obs.max_steps} max steps"]
        self.plot_figs = []
        self.plot_paths = []
        self._trace_steps = []

    def add_log(self, msg: str):
        self.log.append(msg)

    def apply_action(self, action: DataCleaningAction):
        obs_before = self.obs
        self.obs = self.env.step(action)
        r = self.obs.reward
        self.add_log(
            f"**[{len(self.obs.history)}]** `{action.action_type}` → "
            f"{self.obs.last_step_summary}  *(reward {r:+.3f})*"
        )
        # Render plot immediately when declared
        if action.action_type == "plot" and action.x and action.y:
            try:
                df_w = self.env._df.copy()
            except Exception:
                df_w = pd.DataFrame(self.obs.preview)
            fig = _render_plot_fig(df_w, action.plot_type or "scatter", action.x, action.y)
            self.plot_figs.append(fig)
            # save to temp path for gallery
            tmp = tempfile.mktemp(suffix=".png")
            fig.savefig(tmp, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            self.plot_paths.append(tmp)

    def render_table(self) -> str:
        if self.obs is None:
            return "<p style='color:#64748b;padding:16px;'>Reset to load data.</p>"
        df = pd.DataFrame(self.obs.preview)
        return _df_to_html(df)

    def render_issues(self) -> str:
        if self.obs is None:
            return ""
        return _issues_html(self.obs.issues or [])

    def render_log(self) -> str:
        return "\n\n".join(reversed(self.log[-30:]))

    def render_score(self) -> str:
        if self.obs is None:
            return ""
        return _score_html(self.obs.terminal_grader_score, self.obs.done)

    def render_stepinfo(self) -> str:
        if self.obs is None:
            return ""
        hist = len(self.obs.history)
        pct  = int(100 * hist / max(self.obs.max_steps, 1))
        bar  = (
            f"<div style='background:#1e293b;border-radius:4px;height:8px;width:100%;margin:6px 0;'>"
            f"<div style='background:#3b82f6;width:{pct}%;height:100%;border-radius:4px;"
            f"transition:width .3s ease;'></div></div>"
        )
        status = "✅ **Done**" if self.obs.done else "⏳ Running"
        return (
            f"**Step {hist} / {self.obs.max_steps}** &nbsp;·&nbsp; "
            f"Reward: `{self.obs.cumulative_reward:.3f}` &nbsp;·&nbsp; {status}\n\n{bar}"
        )


# ── oracle next-action ─────────────────────────────────────────────────────────
def _oracle_next(sess: Session) -> Optional[DataCleaningAction]:
    from data_cleaning_env.baseline_inference import (
        _history_has_action_type, _fill_missing_count, _hard_pipeline_next_action,
    )
    obs, task = sess.obs, sess.task
    if task == "easy":
        if not _history_has_action_type(obs, "remove_duplicates"):
            return DataCleaningAction(action_type="remove_duplicates")
        if _fill_missing_count(obs) < 1:
            return DataCleaningAction(action_type="fill_missing", column="Price", method="mean")
        return DataCleaningAction(action_type="submit")
    if task == "medium":
        return DataCleaningAction(action_type="compute_metrics") \
            if not _history_has_action_type(obs, "compute_metrics") \
            else DataCleaningAction(action_type="submit")
    if task == "medium_plus":
        return DataCleaningAction(action_type="compute_kpis") \
            if not _history_has_action_type(obs, "compute_kpis") \
            else DataCleaningAction(action_type="submit")
    if task in ("hard", "expert"):
        if not _history_has_action_type(obs, "remove_duplicates"):
            return DataCleaningAction(action_type="remove_duplicates")
        return _hard_pipeline_next_action(obs) or DataCleaningAction(action_type="submit")
    return DataCleaningAction(action_type="submit")


# ── shared render helper ───────────────────────────────────────────────────────
def _pack(sess: Session, gallery_update=None, report_update=None):
    """Pack all output component values."""
    plots = [(p, "") for p in sess.plot_paths] if sess.plot_paths else []
    g_upd = gallery_update if gallery_update is not None else gr.update(value=plots or None)
    r_upd = report_update if report_update is not None else gr.update(visible=False, value=None)
    return (
        sess,
        sess.render_table(),
        sess.render_issues(),
        sess.render_log(),
        sess.render_score(),
        sess.render_stepinfo(),
        g_upd,
        r_upd,
    )


# ── event handlers ─────────────────────────────────────────────────────────────
def do_reset(task, sess):
    sess.reset(task)
    hint = _pipeline_html(task)
    return (sess,
            sess.render_table(), sess.render_issues(), sess.render_log(),
            sess.render_score(), sess.render_stepinfo(),
            gr.update(value=None),          # gallery cleared
            gr.update(visible=False, value=None),  # report hidden
            hint)


def do_oracle_step(sess):
    if sess.obs is None or sess.obs.done:
        sess.add_log("⚠️ Episode done or not started.")
        return _pack(sess)
    a = _oracle_next(sess)
    if a:
        sess.apply_action(a)
    plots = [(p, "") for p in sess.plot_paths] if sess.plot_paths else None
    return _pack(sess, gallery_update=gr.update(value=plots))


def do_oracle_all(sess):
    if sess.obs is None:
        sess.add_log("⚠️ Reset first.")
        return _pack(sess)
    for _ in range(60):
        if sess.obs.done:
            break
        a = _oracle_next(sess)
        if a is None:
            break
        sess.apply_action(a)
    plots = [(p, "") for p in sess.plot_paths] if sess.plot_paths else None
    return _pack(sess, gallery_update=gr.update(value=plots))


def do_manual_step(action_json, sess):
    if sess.obs is None:
        return _pack(sess)
    if sess.obs.done:
        sess.add_log("⚠️ Episode done. Reset to play again.")
        return _pack(sess)
    try:
        data = json.loads(action_json)
        action = DataCleaningAction(**data)
    except Exception as e:
        sess.add_log(f"❌ JSON parse error: {e}")
        return _pack(sess)
    sess.apply_action(action)
    plots = [(p, "") for p in sess.plot_paths] if sess.plot_paths else None
    return _pack(sess, gallery_update=gr.update(value=plots))


def do_download_report(sess):
    """Generate docx and return as a downloadable file."""
    if sess.obs is None:
        return gr.update(value=None, visible=False)
    path = _generate_report(sess)
    if path:
        return gr.update(value=path, visible=True)
    return gr.update(value=None, visible=False)


def do_llm_run(groq_key, model_name, task, max_steps, sess):
    """Stream LLM episode step-by-step."""
    key = (groq_key or "").strip() or os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        sess.add_log("❌ No Groq API key — paste it above or set GROQ_API_KEY env var.")
        yield _pack(sess)
        return
    try:
        from openai import OpenAI
    except ImportError:
        sess.add_log("❌ `openai` package missing — run: pip install openai")
        yield _pack(sess)
        return

    from data_cleaning_env.baseline_inference import (
        _chat_completion_with_retry, _parse_action_json,
        _hard_alternate_loop_normalize, _stuck_advance, _semantic_action_repr,
    )

    sess.reset(task)
    sess.add_log(f"🧠 **LLM mode** — model `{model_name}` · task `{task}`")
    yield _pack(sess)

    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    obs = sess.obs
    limit = min(int(max_steps), obs.max_steps)
    prev_semantic: Optional[str] = None
    HW = 8

    for step_i in range(1, limit + 1):
        if obs.done:
            break
        recent = obs.history[-HW:] if obs.history else []
        hn = f"(last {HW} of {len(obs.history)})" if len(obs.history) > HW else ""
        prompt = f"""You are an e-commerce data analyst cleaning a pandas table.
Task: {task}
Instruction: {obs.instruction}
Columns: {obs.column_names}
Issues: {obs.issues}
Policy rules: {json.dumps(getattr(obs, 'policy_rules', []) or [])}
Policy warnings: {json.dumps(getattr(obs, 'policy_warnings', []) or [])}
Preview: {json.dumps(obs.preview, indent=0)}
Recent history {hn}: {recent}
Last step: {obs.last_step_summary or "(none)"}

Rules:
- Do NOT fill_missing or drop_column on OrderID/CustomerID.
- Easy: remove_duplicates → fill_missing(Price,mean) → submit.
- Medium: compute_metrics → submit.
- Medium_plus: compute_kpis → submit.
- Hard/Expert: remove_duplicates → fill_missing(Price,mean) → derive_revenue → plot scatter(OrderDate,Revenue) → plot bar(Category,Revenue) → submit.
- ExpiryDays can appear in warnings, but do not treat expiry cleanup as a required task step.
- Submit as soon as instruction is satisfied.

Reply ONE JSON only — no markdown fences:
{{"action_type":"...","column":null,"method":null,"z_threshold":3.0,"x":null,"y":null,"plot_type":null,"export_basename":null,"description":"..."}}"""

        sess.add_log(f"🤔 *Step {step_i}: asking LLM…*")
        yield _pack(sess)

        try:
            resp = _chat_completion_with_retry(
                client, model_name,
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_retries=3, initial_delay_s=8,
                json_mode=True,
            )
            raw  = resp.choices[0].message.content or ""
            data = _parse_action_json(raw)
            action = DataCleaningAction.model_validate(data)
            hfix = _hard_alternate_loop_normalize(task, obs, action)
            if hfix:
                action = hfix
            sfix = _stuck_advance(task, obs, action, prev_semantic)
            if sfix:
                action = sfix
        except Exception as e:
            sess.add_log(f"❌ LLM error: {e}")
            action = DataCleaningAction(action_type="noop")

        # Remove "asking LLM" placeholder
        if len(sess.log) >= 2 and "asking LLM" in sess.log[-2]:
            sess.log.pop(-2)

        prev_semantic = _semantic_action_repr(action)
        sess.apply_action(action)
        obs = sess.obs

        plots = [(p, "") for p in sess.plot_paths] if sess.plot_paths else None
        yield _pack(sess, gallery_update=gr.update(value=plots))
        if obs.done:
            break

    if not obs.done:
        sess.apply_action(DataCleaningAction(action_type="submit"))

    plots = [(p, "") for p in sess.plot_paths] if sess.plot_paths else None
    yield _pack(sess, gallery_update=gr.update(value=plots))


def build_json(at, col, method, z, pt, x, y, desc):
    return json.dumps({
        "action_type": at,
        "column":       col.strip() or None,
        "method":       method.strip() or None,
        "z_threshold":  float(z) if z else 3.0,
        "plot_type":    pt.strip() or None,
        "x":            x.strip() or None,
        "y":            y.strip() or None,
        "description":  desc.strip() or None,
        "export_basename": None,
    }, indent=2)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="AutoDataLab — E-commerce Data Analyst") as demo:

    sess_state = gr.State(Session())

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1e40af,#2563eb);color:white;
                border-radius:12px;padding:20px 28px;margin-bottom:16px;">
      <h1 style="margin:0;font-size:26px;">🛒 AutoDataLab — E-commerce Data Analyst</h1>
      <p style="margin:6px 0 0 0;opacity:.9;font-size:14px;">
        RL-style environment · Data cleaning · Business metrics · Visualization
        &nbsp;·&nbsp; Oracle mode (no key) or LLM mode (Groq API key)
      </p>
    </div>
    """)

    # ── top row ────────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            task_dd      = gr.Dropdown(choices=TASKS, value="easy", label="📋 Task")
            task_desc_md = gr.Markdown(TASK_DESCRIPTIONS["easy"])
        with gr.Column(scale=3):
            step_info_md = gr.Markdown("")
            score_html   = gr.HTML("")

    # ── data / log / issues / plots tabs ─────────────────────────────────────
    with gr.Tabs():
        with gr.TabItem("📋 Data Preview"):
            table_html = gr.HTML(
                "<p style='color:#64748b;padding:16px;'>Reset an episode to load data.</p>"
            )
        with gr.TabItem("🚨 Issues"):
            issues_html = gr.HTML("")
        with gr.TabItem("📜 Action Log"):
            log_md = gr.Markdown("", elem_classes=["log-box"])
        with gr.TabItem("📈 Plots"):
            plots_gallery = gr.Gallery(
                label="Declared plots (appear after each plot action)",
                columns=2, rows=1, height="auto", object_fit="contain",
                show_label=True, elem_classes=["plot-gallery"],
            )

    gr.Markdown("---")

    # ── Report download row ────────────────────────────────────────────────────
    with gr.Row():
        download_btn  = gr.Button("📄 Download Word Report (.docx)", variant="secondary", scale=1)
        report_file   = gr.File(label="", visible=False, scale=1)
        with gr.Column(scale=2):
            gr.Markdown(
                "<span style='color:#94a3b8;font-size:13px;'>"
                "Generates a .docx with episode summary, steps, and table preview.</span>"
            )

    gr.Markdown("---")

    # ── Mode tabs ──────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ORACLE ──────────────────────────────────────────────────────────────
        with gr.TabItem("🤖 Oracle Mode"):
            gr.Markdown(
                "The **deterministic expert policy** always picks the right next action. "
                "No API key needed. Great for understanding the pipeline."
            )
            with gr.Row():
                reset_o_btn   = gr.Button("🔄 Reset Episode", variant="primary")
                oracle_1_btn  = gr.Button("▶ Next Step", variant="secondary")
                oracle_all_btn = gr.Button("⚡ Run All Steps", variant="secondary")

            hint_html = gr.HTML(_pipeline_html("easy"))

        # LLM ─────────────────────────────────────────────────────────────────
        with gr.TabItem("🧠 LLM Mode (Groq)"):
            gr.Markdown(
                "Paste your **Groq API key** — the LLM drives the agent step-by-step with live updates. "
                "Free keys at [console.groq.com](https://console.groq.com/keys)."
            )
            with gr.Row():
                groq_key_inp = gr.Textbox(
                    label="🔑 Groq API Key", placeholder="gsk_...", type="password",
                    value=os.environ.get("GROQ_API_KEY", ""), scale=3,
                )
                llm_model_dd = gr.Dropdown(choices=GROQ_MODELS, value=GROQ_MODELS[0],
                                           label="Model", scale=1)
            with gr.Row():
                llm_task_dd    = gr.Dropdown(choices=TASKS, value="easy", label="Task", scale=1)
                llm_steps_sl   = gr.Slider(5, 50, value=30, step=1,
                                           label="Max steps", scale=2)
            llm_run_btn = gr.Button("🚀 Run LLM Episode", variant="primary", size="lg")

        # MANUAL ──────────────────────────────────────────────────────────────
        with gr.TabItem("🎮 Manual Mode"):
            gr.Markdown("Build every action yourself via the form below.")
            reset_m_btn = gr.Button("🔄 Reset Episode", variant="primary")
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=2):
                    at_dd    = gr.Dropdown(
                        choices=["remove_duplicates","fill_missing","drop_column","normalize",
                                 "remove_outliers","derive_revenue","compute_metrics","compute_kpis",
                                 "plot","export_csv","submit","noop"],
                        value="remove_duplicates", label="action_type",
                    )
                    with gr.Row():
                        col_inp  = gr.Textbox(label="column", placeholder="Price", value="")
                        meth_dd  = gr.Dropdown(choices=["","mean","median","mode"],
                                               value="", label="method")
                    with gr.Row():
                        z_inp    = gr.Number(label="z_threshold", value=3.0)
                        pt_dd    = gr.Dropdown(choices=["","scatter","bar","histogram"],
                                               value="", label="plot_type")
                    with gr.Row():
                        x_inp    = gr.Textbox(label="x column", placeholder="OrderDate", value="")
                        y_inp    = gr.Textbox(label="y column", placeholder="Revenue", value="")
                    desc_inp = gr.Textbox(label="description (optional)", value="")
                    exec_btn = gr.Button("▶ Execute Action", variant="primary")
                with gr.Column(scale=1):
                    gr.Markdown("#### 📤 JSON Preview")
                    json_out = gr.Code(language="json", label="", lines=16)

    # ── outputs tuple ──────────────────────────────────────────────────────────
    # (sess, table, issues, log, score, stepinfo, gallery, report_file)
    OUTS = [sess_state, table_html, issues_html, log_md, score_html, step_info_md,
            plots_gallery, report_file]
    # reset also updates hint_html
    OUTS_R = OUTS + [hint_html]

    # ── wiring ─────────────────────────────────────────────────────────────────
    def _upd_task(task):
        return TASK_DESCRIPTIONS[task]
    task_dd.change(_upd_task, [task_dd], [task_desc_md])

    reset_o_btn.click(do_reset, [task_dd, sess_state], OUTS_R)
    reset_m_btn.click(do_reset, [task_dd, sess_state], OUTS_R)

    oracle_1_btn.click(do_oracle_step, [sess_state], OUTS)
    oracle_all_btn.click(do_oracle_all, [sess_state], OUTS)

    form_ins = [at_dd, col_inp, meth_dd, z_inp, pt_dd, x_inp, y_inp, desc_inp]
    for inp in form_ins:
        inp.change(build_json, form_ins, [json_out])
    exec_btn.click(
        fn=lambda *a: do_manual_step(build_json(*a[:-1]), a[-1]),
        inputs=form_ins + [sess_state],
        outputs=OUTS,
    )

    llm_run_btn.click(
        fn=do_llm_run,
        inputs=[groq_key_inp, llm_model_dd, llm_task_dd, llm_steps_sl, sess_state],
        outputs=OUTS,
    )

    download_btn.click(do_download_report, [sess_state], [report_file])

    demo.load(fn=lambda s: do_reset("easy", s), inputs=[sess_state], outputs=OUTS_R)


# ── entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--share",  action="store_true")
    ap.add_argument("--port",   type=int, default=7861)
    ap.add_argument("--host",   default="0.0.0.0")
    ap.add_argument("--hf",     action="store_true", help="HF Spaces mode: mount OpenEnv API + Gradio on same port")
    args = ap.parse_args()

    hf_mode = args.hf or os.environ.get("SPACE_ID") is not None  # auto-detect HF Spaces

    if hf_mode:
        # Mount Gradio UI on top of the OpenEnv FastAPI app so both work on port 7860
        import uvicorn
        from data_cleaning_env.models import DataCleaningAction, DataCleaningObservation
        from data_cleaning_env.server.data_cleaning_env_environment import DataCleaningEnvironment
        try:
            from openenv.core.env_server.http_server import create_app as create_openenv_app
            openenv_fastapi = create_openenv_app(
                DataCleaningEnvironment, DataCleaningAction, DataCleaningObservation,
                env_name="data_cleaning_env",
            )
            # Mount Gradio at /ui; OpenEnv API stays at root (/, /reset, /step, /state, /health)
            combined_app = gr.mount_gradio_app(openenv_fastapi, demo, path="/ui", css=CSS)
            print("Running combined OpenEnv API + Gradio UI on port 7860")
            print("  API:  http://0.0.0.0:7860/health")
            print("  UI:   http://0.0.0.0:7860/ui")
            uvicorn.run(combined_app, host="0.0.0.0", port=7860)
        except Exception as e:
            print(f"[warn] Could not mount OpenEnv app ({e}), falling back to Gradio-only")
            demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
    else:
        demo.launch(server_name=args.host, server_port=args.port,
                    share=args.share, inbrowser=not args.share, css=CSS)
