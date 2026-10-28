"""
Thesis-ready charts and tables (pt-BR, IEEE-like style) for the benchmark results.

Reads the CSVs written by the benchmark scripts into `results/` and writes one chart
per file - PNG (600 dpi) and a vector PDF - into `results/visualizations/academic/`,
plus the tables backing them (CSV + LaTeX/booktabs) into `results/visualizations/tables/`.

Inputs (all optional except where noted; missing inputs skip their charts):
  results/ingestion.csv                  ingestion_time.py (one row per video)
  results/search_latency_<label>.csv     search_latency.py
  results/resources.csv (+ _host.csv)    monitor_resources.sh during ingestion
  results/resources_search.csv           monitor_resources.sh during the search run
  results/storage.csv                    storage_footprint.sh (one row per hour)
  results/invalid_cache_leak/            run with the llama.cpp prompt-cache leak (before/after charts)

Styling follows model-test-suite/src/visualize_academic.py: white background, serif font
(Times New Roman when installed, else STIXGeneral), Okabe-Ito colorblind-safe colors,
pt-BR number formatting (decimal comma, dot as thousands separator).

Usage:
    python benchmarks/visualize_academic.py [--results results] [--search-label idle]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("visualize_academic")
logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font subsetting is very chatty at INFO

# Containers of the system, largest memory consumers first. Anything else running on
# the host (e.g. unrelated containers) is ignored.
CONTAINER_LABELS = {
    "qwen-api": "llama.cpp (Qwen3.5 0.8B)",
    "clip-api": "CLIP API",
    "guard-worker": "Worker de pré-processamento",
    "guard-enrichment-worker": "Worker de enriquecimento",
    "pi-guard": "Pi Guard (API)",
    "chromadb": "ChromaDB",
    "watcher": "Watcher",
    "redis": "Redis",
}
INFRA_CONTAINERS = ("chromadb", "watcher", "redis")
INFRA_LABEL = "Infraestrutura (ChromaDB, Watcher, Redis)"

# Okabe-Ito, one color per service; the small infra containers share a neutral gray.
# Qwen keeps the green it has in model-test-suite so both chapters read the same.
CONTAINER_COLORS = {
    "qwen-api": "#009E73",
    "clip-api": "#0072B2",
    "guard-worker": "#E69F00",
    "guard-enrichment-worker": "#D55E00",
    "pi-guard": "#CC79A7",
    "infra": "#999999",
}
STAGE_COLORS = {"stage1": "#0072B2", "stage2": "#E69F00"}
SERIES_COLORS = {"primary": "#0072B2", "secondary": "#D55E00", "reference": "black"}
STAGE2_HATCH = "////"

FIG_WIDTH_IN = 3.5  # IEEE single column
WIDE_FIG_WIDTH_IN = 7.16  # IEEE double column
BAR_FIG_SIZE = (FIG_WIDTH_IN, 2.5)
TALL_BAR_FIG_SIZE = (FIG_WIDTH_IN, 3.0)
LINE_FIG_SIZE = (FIG_WIDTH_IN, 2.7)
TIMELINE_FIG_SIZE = (WIDE_FIG_WIDTH_IN, 3.0)
DPI = 600
VALUE_AXIS_HEADROOM = 1.18
MIB_PER_GIB = 1024
BYTES_PER_MIB = 1024 ** 2
BYTES_PER_GIB = 1024 ** 3
PI5_THROTTLE_TEMP_C = 80  # Raspberry Pi 5 firmware starts throttling the ARM cores at 80 °C


# --------------------------------------------------------------------------- #
# Style & formatting helpers
# --------------------------------------------------------------------------- #

def apply_academic_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": "#b0b0b0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "hatch.color": "black",
        "hatch.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,  # embed TrueType fonts (editable text, accepted by IEEE PDF checks)
    })


def fmt_br(value: float, decimals: int = 2, suffix: str = "") -> str:
    """pt-BR number: decimal comma, dot as thousands separator (e.g. 5.504,2)."""
    text = f"{value:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"{text}{suffix}"


def br_tick_formatter(suffix: str = "") -> FuncFormatter:
    def _format(value, _pos):
        if math.isclose(value, round(value), abs_tol=1e-9):
            return fmt_br(round(value), 0, suffix)
        decimals = 1 if math.isclose(value * 10, round(value * 10), abs_tol=1e-9) else 2
        return fmt_br(value, decimals, suffix)
    return FuncFormatter(_format)


def legend_below(ax, handles=None, ncol: int = 2, offset: float = -0.22):
    kwargs = {"handles": handles} if handles is not None else {}
    return ax.legend(**kwargs, loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol,
                     frameon=True, fancybox=False, edgecolor="black", framealpha=1.0,
                     handletextpad=0.4, columnspacing=1.0)


# --------------------------------------------------------------------------- #
# Saving & overlap check
# --------------------------------------------------------------------------- #

def _bbox_overlap(a, b, tolerance_px: float = 1.0) -> bool:
    return (a.x0 + tolerance_px < b.x1 and b.x0 + tolerance_px < a.x1
            and a.y0 + tolerance_px < b.y1 and b.y0 + tolerance_px < a.y1)


def _warn_overlaps(fig, name: str) -> None:
    """Log a warning if any visible text overlaps another text, a bar, or spills out of its axes."""
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        texts = [t for t in ax.texts if t.get_visible() and t.get_text()]
        labels = [t for t in (*ax.get_xticklabels(), *ax.get_yticklabels(),
                              ax.xaxis.label, ax.yaxis.label, ax.title, *fig.texts)
                  if t.get_visible() and t.get_text()]
        boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts + labels]
        legend = ax.get_legend()
        if legend is not None:
            boxes.append(("<legend>", legend.get_window_extent(renderer)))

        for (text_a, box_a), (text_b, box_b) in itertools.combinations(boxes, 2):
            if _bbox_overlap(box_a, box_b):
                log.warning("[%s] text overlap: %r <-> %r", name, text_a, text_b)

        axes_box = ax.get_window_extent(renderer)
        bar_boxes = [p.get_window_extent(renderer) for p in ax.patches]
        for text in texts:
            box = text.get_window_extent(renderer)
            if any(_bbox_overlap(box, bar) for bar in bar_boxes):
                log.warning("[%s] annotation %r overlaps a bar", name, text.get_text())
            if box.x0 < axes_box.x0 - 1 or box.x1 > axes_box.x1 + 1:
                log.warning("[%s] annotation %r extends outside the plot area", name, text.get_text())


def save(fig, out_dir: Path, stem: str) -> None:
    fig.canvas.draw()
    _warn_overlaps(fig, stem)
    for ext, kwargs in (("png", {"dpi": DPI}), ("pdf", {})):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)
    log.info("Saved %s.{png,pdf}", out_dir / stem)


def _fit_labels_in_axes(fig, ax, annotations, axis: str = "x", margin_px: float = 4.0) -> None:
    """Widen the value axis until every bar label is fully inside the plot area."""
    for _ in range(5):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_box = ax.get_window_extent(renderer)
        extents = [a.get_window_extent(renderer) for a in annotations if a.get_text()]
        if not extents:
            return
        if axis == "x":
            overflow = max(e.x1 for e in extents) - (axes_box.x1 - margin_px)
            span_px = axes_box.width
            low, high = ax.get_xlim()
        else:
            overflow = max(e.y1 for e in extents) - (axes_box.y1 - margin_px)
            span_px = axes_box.height
            low, high = ax.get_ylim()
        if overflow <= 0:
            return
        high += overflow * (high - low) / span_px * 1.1
        (ax.set_xlim if axis == "x" else ax.set_ylim)(low, high)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def _latex_escape(text: str) -> str:
    for char, escaped in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                          ("#", r"\#"), ("$", r"\$"), ("±", r"$\pm$"), ("×", r"$\times$")):
        text = text.replace(char, escaped)
    return text


def write_table(table: pd.DataFrame, display: pd.DataFrame, out_dir: Path, stem: str,
                caption: str, label: str, midrule_before_last: bool = False) -> None:
    """`table` (raw numbers) -> CSV; `display` (pt-BR formatted strings) -> booktabs LaTeX."""
    table.to_csv(out_dir / f"{stem}.csv", index=False)

    colspec = "l" + "r" * (len(display.columns) - 1)
    rows = [" & ".join(_latex_escape(str(v)) for v in row) + r" \\" for row in display.itertuples(index=False)]
    if midrule_before_last and len(rows) > 1:
        rows.insert(len(rows) - 1, r"\midrule")

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{_latex_escape(caption)}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{colspec}}}",
        r"    \toprule",
        "    " + " & ".join(_latex_escape(c) for c in display.columns) + r" \\",
        r"    \midrule",
        *(f"    {row}" for row in rows),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    (out_dir / f"{stem}.tex").write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved %s.{csv,tex}", out_dir / stem)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_resources(path: Path, start: float | None = None, end: float | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["container"].isin(CONTAINER_LABELS)]
    if start is not None:
        df = df[df["epoch"] >= start]
    if end is not None:
        df = df[df["epoch"] <= end]
    return df


def load_host(path: Path, start: float | None = None, end: float | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if start is not None:
        df = df[df["epoch"] >= start]
    if end is not None:
        df = df[df["epoch"] <= end]
    return df


def container_group(container: str) -> str:
    return "infra" if container in INFRA_CONTAINERS else container


def group_label(group: str) -> str:
    return INFRA_LABEL if group == "infra" else CONTAINER_LABELS[group]


def latency_stats(latencies_s: list[float]) -> dict:
    cuts = statistics.quantiles(latencies_s, n=100, method="inclusive")
    return {
        "n": len(latencies_s),
        "mean": statistics.fmean(latencies_s),
        "stdev": statistics.stdev(latencies_s),
        "median": statistics.median(latencies_s),
        "min": min(latencies_s),
        "max": max(latencies_s),
        "p90": cuts[89],
        "p99": cuts[98],
    }


# --------------------------------------------------------------------------- #
# Charts - ingestion
# --------------------------------------------------------------------------- #

def plot_ingestion_time(ingestion: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar per video: stage 1 + the stage 2 tail that runs after it = end-to-end time."""
    fig, ax = plt.subplots(figsize=BAR_FIG_SIZE, layout="constrained")
    positions = list(range(len(ingestion)))
    stage1 = ingestion["stage1_s"] / 60
    stage2 = ingestion["stage2_tail_s"] / 60
    total = ingestion["total_s"] / 60

    ax.barh(positions, stage1, height=0.6, color=STAGE_COLORS["stage1"], edgecolor="black",
            linewidth=0.6, zorder=2, label="Estágio 1 (amostragem + CLIP)")
    ax.barh(positions, stage2, left=stage1, height=0.6, color=STAGE_COLORS["stage2"], hatch=STAGE2_HATCH,
            edgecolor="black", linewidth=0.6, zorder=2, label="Estágio 2 (YOLO + VLM), após o Estágio 1")
    video_minutes = ingestion["video_duration_s"].astype(float).mean() / 60
    ax.axvline(video_minutes, color="black", linestyle=":", linewidth=1.0, zorder=3,
               label=f"Duração do vídeo ({fmt_br(video_minutes, 0)} min)")

    # Totals sit in one column past both the longest bar and the video-duration line, so no label crosses it.
    label_x = max(total.max(), video_minutes) + total.max() * 0.02
    annotations = [ax.text(label_x, p, fmt_br(t, 1, " min"), va="center", fontsize=7.5)
                   for p, t in zip(positions, total)]

    ax.set_yticks(positions)
    ax.set_yticklabels([f"Vídeo {i + 1}" for i in positions])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(total.max(), video_minutes) * VALUE_AXIS_HEADROOM)
    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="x")
    ax.set_xlabel("Tempo desde a detecção pelo Watcher (min)")
    fig.suptitle("Ingestão ponta a ponta de vídeos de 60 min", fontsize=plt.rcParams["axes.titlesize"])
    handles, _ = ax.get_legend_handles_labels()
    legend_below(ax, handles=handles[1:] + handles[:1], ncol=1, offset=-0.25)  # stages first, then the reference line
    _fit_labels_in_axes(fig, ax, annotations)
    save(fig, out_dir, "ingestion_time_per_video")


def plot_ingestion_vs_events(ingestion: pd.DataFrame, out_dir: Path) -> None:
    """End-to-end time vs. number of events: the VLM cost per event drives the total."""
    fig, ax = plt.subplots(figsize=LINE_FIG_SIZE, layout="constrained")
    events = ingestion["events"].astype(float)
    total_min = ingestion["total_s"] / 60

    # The fit line rises to the right, so labels go below-right of each point; when two points are
    # close in x, the left one is labelled on its left instead (above the line there).
    order = events.sort_values().index.to_list()
    for i, (x, y) in enumerate(zip(events, total_min)):
        ax.scatter(x, y, s=46, marker="o", color=SERIES_COLORS["primary"], edgecolor="black", linewidth=0.6, zorder=3)
        rank = order.index(events.index[i])
        crowded_left = rank + 1 < len(order) and events[order[rank + 1]] - x < events.max() * 0.15
        offset, ha = ((-7, 0), "right") if crowded_left else ((6, -9), "left")
        ax.annotate(f"Vídeo {i + 1}", (x, y), textcoords="offset points", xytext=offset, ha=ha,
                    va="center", fontsize=7.5)

    # Least-squares line through the origin: minutes per event.
    slope = float((events * total_min).sum() / (events ** 2).sum())
    xs = np.linspace(0, events.max() * 1.15, 50)
    ax.plot(xs, slope * xs, color=SERIES_COLORS["reference"], linestyle="--", linewidth=1.0, zorder=2,
            label=f"Ajuste: {fmt_br(slope * 60, 1)} s por evento")

    ax.set_xlim(0, events.max() * 1.15)
    ax.set_ylim(0, total_min.max() * 1.2)
    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.yaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="both")
    ax.set_xlabel("Eventos detectados no vídeo")
    ax.set_ylabel("Tempo de ingestão (min)")
    fig.suptitle("Tempo de ingestão × número de eventos", fontsize=plt.rcParams["axes.titlesize"])
    legend_below(ax, ncol=1)
    save(fig, out_dir, "ingestion_time_vs_events")


def table_ingestion(ingestion: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for i, r in ingestion.reset_index(drop=True).iterrows():
        rows.append({
            "video": f"Vídeo {i + 1}",
            "stage1_min": r["stage1_s"] / 60,
            "stage2_tail_min": r["stage2_tail_s"] / 60,
            "total_min": r["total_s"] / 60,
            "frames": int(r["frames"]),
            "events": int(r["events"]),
            "dropped_events": int(r["dropped_events"]),
            "seconds_per_event": r["total_s"] / r["events"],
            "realtime_factor": r["realtime_factor"],
        })
    raw = pd.DataFrame(rows)

    numeric = raw.drop(columns="video")
    mean, sd = numeric.mean(), numeric.std()
    raw_with_summary = pd.concat([raw, pd.DataFrame([{"video": "Média", **mean}, {"video": "Desvio-padrão", **sd}])],
                                 ignore_index=True)

    def fmt_row(label, values, spread=None):
        def cell(col, decimals):
            text = fmt_br(values[col], decimals)
            return f"{text} ± {fmt_br(spread[col], decimals)}" if spread is not None else text
        return [label, cell("stage1_min", 1), cell("stage2_tail_min", 1), cell("total_min", 1),
                cell("events", 0), cell("seconds_per_event", 1), cell("realtime_factor", 2)]

    display_rows = [fmt_row(r["video"], r) for _, r in raw.iterrows()]
    display_rows.append(fmt_row("Média ± DP", mean, sd))
    display = pd.DataFrame(display_rows, columns=[
        "Vídeo", "Estágio 1 (min)", "Estágio 2, restante (min)", "Total (min)", "Eventos",
        "s/evento", "Fator de tempo real",
    ])
    write_table(raw_with_summary, display, out_dir, "ingestion",
                "Tempo de ingestão ponta a ponta de vídeos de 60 min no Raspberry Pi 5",
                "tab:ingestao", midrule_before_last=True)


# --------------------------------------------------------------------------- #
# Charts - search latency
# --------------------------------------------------------------------------- #

def plot_search_latency(latencies_s: list[float], stats: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=LINE_FIG_SIZE, layout="constrained")
    bin_width = 0.2
    bins = np.arange(math.floor(min(latencies_s) / bin_width) * bin_width,
                     max(latencies_s) + bin_width, bin_width)
    ax.hist(latencies_s, bins=bins, color=SERIES_COLORS["primary"], edgecolor="black", linewidth=0.6, zorder=2)

    markers = [("Média", stats["mean"], "-"), ("p90", stats["p90"], "--"), ("p99", stats["p99"], ":")]
    handles = [Line2D([], [], color="#0072B2", marker="s", linestyle="none", markersize=6,
                      markeredgecolor="black", markeredgewidth=0.6, label=f"Requisições (n = {stats['n']})")]
    for name, value, style in markers:
        ax.axvline(value, color="black", linestyle=style, linewidth=1.1, zorder=3)
        handles.append(Line2D([], [], color="black", linestyle=style, linewidth=1.1,
                              label=f"{name}: {fmt_br(value, 2, ' s')}"))

    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.yaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="y")
    ax.set_xlabel("Tempo de resposta da busca (s)")
    ax.set_ylabel("Número de requisições")
    fig.suptitle("Latência de busca textual com reranking", fontsize=plt.rcParams["axes.titlesize"])
    legend_below(ax, handles=handles, ncol=2)
    save(fig, out_dir, "search_latency_distribution")


def table_search(stats: dict, out_dir: Path) -> None:
    raw = pd.DataFrame([{key: stats[key] for key in ("n", "mean", "stdev", "median", "min", "max", "p90", "p99")}])
    display = pd.DataFrame([[
        str(stats["n"]), *(fmt_br(stats[k], 2) for k in ("mean", "stdev", "median", "min", "max", "p90", "p99")),
    ]], columns=["n", "Média (s)", "DP (s)", "Mediana (s)", "Mín. (s)", "Máx. (s)", "p90 (s)", "p99 (s)"])
    write_table(raw, display, out_dir, "search_latency",
                "Latência de ponta a ponta da busca textual (CLIP + ChromaDB + reranking no llama.cpp)",
                "tab:latencia-busca")


# --------------------------------------------------------------------------- #
# Charts - resources
# --------------------------------------------------------------------------- #

def peak_per_container(resources: pd.DataFrame) -> pd.DataFrame:
    peaks = resources.groupby("container").agg(peak_mem_mib=("mem_used_mib", "max"),
                                               peak_cpu_percent=("cpu_percent", "max"))
    return peaks.reindex([c for c in CONTAINER_LABELS if c in peaks.index])


def plot_peak_hbar(values: pd.Series, title: str, xlabel: str, stem: str, out_dir: Path,
                   decimals: int = 0, suffix: str = "", xlim_max: float | None = None) -> None:
    containers = list(values.index)
    fig, ax = plt.subplots(figsize=TALL_BAR_FIG_SIZE, layout="constrained")
    positions = range(len(containers))
    bars = ax.barh(positions, values.to_list(), height=0.62,
                   color=[CONTAINER_COLORS[container_group(c)] for c in containers],
                   edgecolor="black", linewidth=0.6, zorder=2)
    annotations = ax.bar_label(bars, labels=[fmt_br(v, decimals, suffix) for v in values], padding=3, fontsize=7.5)

    ax.set_yticks(list(positions))
    ax.set_yticklabels([CONTAINER_LABELS[c] for c in containers])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xlim_max if xlim_max is not None else values.max() * VALUE_AXIS_HEADROOM)
    ax.xaxis.set_major_formatter(br_tick_formatter(suffix.strip() if suffix.strip() == "%" else ""))
    ax.grid(axis="x")
    ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=plt.rcParams["axes.titlesize"])
    _fit_labels_in_axes(fig, ax, annotations)
    save(fig, out_dir, stem)


def _video_markers(ax, ingestion: pd.DataFrame, origin: float) -> None:
    """Dotted line + label at the start of each video, in hours since `origin`."""
    for i, t0 in enumerate(ingestion["t0"]):
        x = (t0 - origin) / 3600
        if i > 0:
            ax.axvline(x, color="black", linestyle=":", linewidth=0.8, zorder=3)
        ax.text(x + 0.02, 0.97, f"Vídeo {i + 1}", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=7.5)


def plot_ram_timeline(resources: pd.DataFrame, host: pd.DataFrame | None, ingestion: pd.DataFrame,
                      out_dir: Path) -> None:
    """Stacked RAM per service over the whole ingestion run, with the Pi's physical RAM as reference."""
    origin = float(ingestion["t0"].iloc[0])
    frame = resources.assign(group=resources["container"].map(container_group))
    pivot = (frame.groupby(["epoch", "group"])["mem_used_mib"].sum().unstack("group").fillna(0.0) / MIB_PER_GIB)
    groups = [g for g in (*[c for c in CONTAINER_LABELS if c not in INFRA_CONTAINERS], "infra") if g in pivot.columns]
    hours = (pivot.index.to_numpy(dtype=float) - origin) / 3600

    fig, ax = plt.subplots(figsize=TIMELINE_FIG_SIZE, layout="constrained")
    ax.stackplot(hours, [pivot[g].to_numpy() for g in groups], colors=[CONTAINER_COLORS[g] for g in groups],
                 labels=[group_label(g) for g in groups], edgecolor="white", linewidth=0.3, zorder=2)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CONTAINER_COLORS[g], edgecolor="black", linewidth=0.6,
                             label=group_label(g)) for g in groups]
    if host is not None and not host.empty:
        total_gib = host["host_mem_total_mib"].iloc[0] / MIB_PER_GIB
        ax.axhline(total_gib, color="black", linestyle="--", linewidth=1.0, zorder=3)
        handles.append(Line2D([], [], color="black", linestyle="--", linewidth=1.0,
                              label=f"RAM física do Raspberry Pi 5 ({fmt_br(total_gib, 1)} GiB)"))
        ax.set_ylim(0, total_gib * 1.12)

    _video_markers(ax, ingestion, origin)
    ax.set_xlim(hours.min(), hours.max())
    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.yaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="y")
    ax.set_xlabel("Tempo desde o início da ingestão (h)")
    ax.set_ylabel("RAM por contêiner (GiB)")
    fig.suptitle("Consumo de memória dos contêineres durante a ingestão de 3 h de vídeo",
                 fontsize=plt.rcParams["axes.titlesize"])
    legend_below(ax, handles=handles, ncol=3, offset=-0.2)
    save(fig, out_dir, "resource_ram_timeline")


def plot_cpu_timeline(resources: pd.DataFrame, ingestion: pd.DataFrame, out_dir: Path) -> None:
    """Stacked CPU per service (1-minute moving average), 400% = the Pi 5's four cores."""
    origin = float(ingestion["t0"].iloc[0])
    frame = resources.assign(group=resources["container"].map(container_group))
    pivot = frame.groupby(["epoch", "group"])["cpu_percent"].sum().unstack("group").fillna(0.0)
    pivot = pivot.rolling(30, min_periods=1).mean()  # 30 samples x 2 s = 1 min
    groups = [g for g in (*[c for c in CONTAINER_LABELS if c not in INFRA_CONTAINERS], "infra") if g in pivot.columns]
    hours = (pivot.index.to_numpy(dtype=float) - origin) / 3600

    fig, ax = plt.subplots(figsize=TIMELINE_FIG_SIZE, layout="constrained")
    ax.stackplot(hours, [pivot[g].to_numpy() for g in groups], colors=[CONTAINER_COLORS[g] for g in groups],
                 edgecolor="white", linewidth=0.3, zorder=2)
    ax.axhline(400, color="black", linestyle="--", linewidth=1.0, zorder=3)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CONTAINER_COLORS[g], edgecolor="black", linewidth=0.6,
                             label=group_label(g)) for g in groups]
    handles.append(Line2D([], [], color="black", linestyle="--", linewidth=1.0, label="Capacidade total (4 núcleos)"))

    _video_markers(ax, ingestion, origin)
    ax.set_xlim(hours.min(), hours.max())
    ax.set_ylim(0, 450)
    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.yaxis.set_major_formatter(br_tick_formatter("%"))
    ax.grid(axis="y")
    ax.set_xlabel("Tempo desde o início da ingestão (h)")
    ax.set_ylabel("Uso de CPU (%, média móvel de 1 min)")
    fig.suptitle("Uso de CPU dos contêineres durante a ingestão de 3 h de vídeo",
                 fontsize=plt.rcParams["axes.titlesize"])
    legend_below(ax, handles=handles, ncol=3, offset=-0.2)
    save(fig, out_dir, "resource_cpu_timeline")


def plot_temperature_timeline(host: pd.DataFrame, ingestion: pd.DataFrame, out_dir: Path) -> None:
    origin = float(ingestion["t0"].iloc[0])
    hours = (host["epoch"].to_numpy(dtype=float) - origin) / 3600
    temp = host["cpu_temp_c"].astype(float).rolling(30, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=TIMELINE_FIG_SIZE, layout="constrained")
    ax.plot(hours, temp, color=SERIES_COLORS["secondary"], linewidth=1.2, zorder=2)
    ax.axhline(PI5_THROTTLE_TEMP_C, color="black", linestyle="--", linewidth=1.0, zorder=3)
    handles = [
        Line2D([], [], color=SERIES_COLORS["secondary"], linewidth=1.2,
               label=f"Temperatura da CPU (média móvel de 1 min; pico de {fmt_br(host['cpu_temp_c'].max(), 1)} °C)"),
        Line2D([], [], color="black", linestyle="--", linewidth=1.0,
               label=f"Início do throttling térmico ({PI5_THROTTLE_TEMP_C} °C)"),
    ]

    _video_markers(ax, ingestion, origin)
    ax.set_xlim(hours.min(), hours.max())
    ax.set_ylim(min(40, temp.min() - 5), PI5_THROTTLE_TEMP_C + 10)
    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.yaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="y")
    ax.set_xlabel("Tempo desde o início da ingestão (h)")
    ax.set_ylabel("Temperatura (°C)")
    fig.suptitle("Temperatura da CPU durante a ingestão de 3 h de vídeo", fontsize=plt.rcParams["axes.titlesize"])
    legend_below(ax, handles=handles, ncol=2, offset=-0.2)
    save(fig, out_dir, "resource_temperature_timeline")


def plot_cache_leak_comparison(before: pd.DataFrame, after: pd.DataFrame, host_before: pd.DataFrame | None,
                               host_after: pd.DataFrame | None, out_dir: Path) -> None:
    """llama.cpp memory with the default prompt cache vs. with `--cache-ram 0`, same workload."""
    fig, ax = plt.subplots(figsize=LINE_FIG_SIZE, layout="constrained")
    for frame, color, style, label in (
        (before, SERIES_COLORS["secondary"], "-", "Cache de prompts padrão"),
        (after, SERIES_COLORS["primary"], "-", "Com --cache-ram 0"),
    ):
        qwen = frame[frame["container"] == "qwen-api"]
        hours = (qwen["epoch"] - qwen["epoch"].iloc[0]) / 3600
        ax.plot(hours, qwen["mem_used_mib"] / MIB_PER_GIB, color=color, linestyle=style, linewidth=1.3,
                zorder=3, label=label)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.yaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="both")
    ax.set_xlabel("Tempo desde o início da ingestão (h)")
    ax.set_ylabel("RAM do llama.cpp (GiB)")
    fig.suptitle("Memória do llama.cpp: cache de prompts em RAM", fontsize=plt.rcParams["axes.titlesize"])
    legend_below(ax, ncol=1)
    save(fig, out_dir, "llama_cpp_cache_ram_comparison")

    if host_before is None or host_after is None:
        return

    fig, ax = plt.subplots(figsize=LINE_FIG_SIZE, layout="constrained")
    for frame, color, label in ((host_before, SERIES_COLORS["secondary"], "Cache de prompts padrão"),
                                (host_after, SERIES_COLORS["primary"], "Com --cache-ram 0")):
        hours = (frame["epoch"] - frame["epoch"].iloc[0]) / 3600
        ax.plot(hours, frame["host_mem_used_mib"] / MIB_PER_GIB, color=color, linewidth=1.3, zorder=3, label=label)
    total_gib = host_after["host_mem_total_mib"].iloc[0] / MIB_PER_GIB
    ax.axhline(total_gib, color="black", linestyle="--", linewidth=1.0, zorder=2,
               label=f"RAM física ({fmt_br(total_gib, 1)} GiB)")

    ax.set_xlim(0, None)
    ax.set_ylim(0, total_gib * 1.12)
    ax.xaxis.set_major_formatter(br_tick_formatter())
    ax.yaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="both")
    ax.set_xlabel("Tempo desde o início da ingestão (h)")
    ax.set_ylabel("RAM usada no host (GiB)")
    fig.suptitle("RAM do Raspberry Pi 5: cache de prompts em RAM", fontsize=plt.rcParams["axes.titlesize"])
    legend_below(ax, ncol=1)
    save(fig, out_dir, "host_ram_cache_comparison")


def table_resources(ingest_peaks: pd.DataFrame, search_peaks: pd.DataFrame | None,
                    host_ingest: pd.DataFrame | None, host_search: pd.DataFrame | None, out_dir: Path) -> None:
    raw = ingest_peaks.rename(columns={"peak_mem_mib": "ingestion_peak_mem_mib",
                                       "peak_cpu_percent": "ingestion_peak_cpu_percent"})
    if search_peaks is not None:
        raw = raw.join(search_peaks.rename(columns={"peak_mem_mib": "search_peak_mem_mib",
                                                    "peak_cpu_percent": "search_peak_cpu_percent"}))
    raw = raw.reset_index().rename(columns={"index": "container"})
    raw.insert(1, "label", raw["container"].map(CONTAINER_LABELS))

    host_row = {"container": "host", "label": "Host (RAM usada no sistema)"}
    if host_ingest is not None:
        host_row["ingestion_peak_mem_mib"] = host_ingest["host_mem_used_mib"].max()
    if host_search is not None:
        host_row["search_peak_mem_mib"] = host_search["host_mem_used_mib"].max()
    raw = pd.concat([raw, pd.DataFrame([host_row])], ignore_index=True)

    def cell(value, decimals, suffix=""):
        return "—" if pd.isna(value) else fmt_br(value, decimals, suffix)

    columns = [("ingestion_peak_mem_mib", "RAM na ingestão (MiB)", 0, ""),
               ("ingestion_peak_cpu_percent", "CPU na ingestão (%)", 0, "")]
    if search_peaks is not None:
        columns += [("search_peak_mem_mib", "RAM na busca (MiB)", 0, ""),
                    ("search_peak_cpu_percent", "CPU na busca (%)", 0, "")]
    display = pd.DataFrame(
        [[r["label"], *(cell(r.get(col), d, s) for col, _, d, s in columns)] for _, r in raw.iterrows()],
        columns=["Contêiner", *(name for _, name, _, _ in columns)],
    )
    write_table(raw, display, out_dir, "resource_peaks",
                "Pico de RAM e CPU por contêiner (100% de CPU = 1 núcleo; máximo de 400% no Raspberry Pi 5)",
                "tab:recursos", midrule_before_last=True)


# --------------------------------------------------------------------------- #
# Charts - storage
# --------------------------------------------------------------------------- #

def plot_storage(storage: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=BAR_FIG_SIZE, layout="constrained")
    hours = storage["hours"].astype(int).to_list()
    total_gib = storage["total_bytes"] / BYTES_PER_GIB
    index_mib = (storage["chroma_bytes"] + storage["sqlite_bytes"]) / BYTES_PER_MIB

    bars = ax.bar(range(len(hours)), total_gib, width=0.6, color=SERIES_COLORS["primary"], edgecolor="black",
                  linewidth=0.6, zorder=2)
    labels = [f"{fmt_br(t, 2, ' GiB')}\níndices: {fmt_br(i, 1, ' MiB')}" for t, i in zip(total_gib, index_mib)]
    annotations = ax.bar_label(bars, labels=labels, padding=3, fontsize=7.5)

    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h} h" for h in hours])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, total_gib.max() * 1.35)
    ax.yaxis.set_major_formatter(br_tick_formatter())
    ax.grid(axis="y")
    ax.set_xlabel("Horas de gravação processadas")
    ax.set_ylabel("Armazenamento ocupado (GiB)")
    fig.suptitle("Crescimento do armazenamento em disco", fontsize=plt.rcParams["axes.titlesize"])
    _fit_labels_in_axes(fig, ax, annotations, axis="y")
    save(fig, out_dir, "storage_growth")


def table_storage(storage: pd.DataFrame, out_dir: Path) -> None:
    raw = storage[["hours", "video_files", "videos_bytes", "chroma_bytes", "sqlite_bytes", "total_bytes",
                   "bytes_per_hour"]].copy()
    display = pd.DataFrame([[
        f"{int(r['hours'])} h",
        fmt_br(r["videos_bytes"] / BYTES_PER_MIB, 1),
        fmt_br(r["chroma_bytes"] / BYTES_PER_MIB, 2),
        fmt_br(r["sqlite_bytes"] / 1024, 1),
        fmt_br(r["total_bytes"] / BYTES_PER_MIB, 1),
        fmt_br(r["bytes_per_hour"] / BYTES_PER_MIB, 1),
    ] for _, r in raw.iterrows()], columns=[
        "Gravação", "Vídeos (MiB)", "ChromaDB (MiB)", "SQLite (KiB)", "Total (MiB)", "Média por hora (MiB/h)",
    ])
    write_table(raw, display, out_dir, "storage",
                "Armazenamento em disco ocupado por horas de gravação processadas", "tab:armazenamento")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis-ready charts and tables for the benchmark results")
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument("--search-label", default="idle")
    args = parser.parse_args()

    results = Path(args.results)
    academic_dir = results / "visualizations" / "academic"
    tables_dir = results / "visualizations" / "tables"
    academic_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    apply_academic_style()

    ingestion = pd.read_csv(results / "ingestion.csv") if (results / "ingestion.csv").exists() else None
    window = None
    if ingestion is not None and not ingestion.empty:
        plot_ingestion_time(ingestion, academic_dir)
        plot_ingestion_vs_events(ingestion, academic_dir)
        table_ingestion(ingestion, tables_dir)
        # Resource charts cover the ingestion run only: from just before the first detection up to the last
        # description, never past it (whatever ran next - e.g. the search benchmark - must not leak in).
        # `docker stats` reports CPU over the ~2 s *after* a sample's timestamp, hence the 5 s margin.
        window = (float(ingestion["t0"].min()) - 60, float(ingestion["t2"].max()) - 5)
    else:
        log.warning("results/ingestion.csv not found; skipping ingestion and resource timelines")

    search_csv = results / f"search_latency_{args.search_label}.csv"
    if search_csv.exists():
        search = pd.read_csv(search_csv)
        latencies_s = (search.loc[search["error"].isna(), "latency_ms"] / 1000).to_list()
        stats = latency_stats(latencies_s)
        plot_search_latency(latencies_s, stats, academic_dir)
        table_search(stats, tables_dir)
    else:
        log.warning("%s not found; skipping search latency", search_csv)

    if window is not None:
        resources = load_resources(results / "resources.csv", *window)
        host = load_host(results / "resources_host.csv", *window)
        if resources is not None and not resources.empty:
            peaks = peak_per_container(resources)
            plot_peak_hbar(peaks["peak_mem_mib"], "Pico de memória por contêiner (ingestão)",
                           "RAM máxima (MiB)", "resource_peak_ram", academic_dir)
            plot_peak_hbar(peaks["peak_cpu_percent"], "Pico de CPU por contêiner (ingestão)",
                           "CPU máxima (%; 100% = 1 núcleo, máx. 400%)", "resource_peak_cpu", academic_dir,
                           suffix="%", xlim_max=400)
            plot_ram_timeline(resources, host, ingestion, academic_dir)
            plot_cpu_timeline(resources, ingestion, academic_dir)
            if host is not None and not host.empty and host["cpu_temp_c"].notna().any():
                plot_temperature_timeline(host, ingestion, academic_dir)

            search_resources = load_resources(results / "resources_search.csv")
            search_host = load_host(results / "resources_search_host.csv")
            search_peaks = peak_per_container(search_resources) if search_resources is not None else None
            table_resources(peaks, search_peaks, host, search_host, tables_dir)

            leak_dir = results / "invalid_cache_leak"
            leak_resources = load_resources(leak_dir / "resources.csv")
            if leak_resources is not None and not leak_resources.empty:
                plot_cache_leak_comparison(leak_resources, resources, load_host(leak_dir / "resources_host.csv"),
                                           host, academic_dir)

    storage_csv = results / "storage.csv"
    if storage_csv.exists():
        storage = pd.read_csv(storage_csv)
        plot_storage(storage, academic_dir)
        table_storage(storage, tables_dir)
    else:
        log.warning("%s not found; skipping storage", storage_csv)

    log.info("Charts saved to %s; tables to %s", academic_dir, tables_dir)


if __name__ == "__main__":
    main()
