"""Post-collection analysis & figure pipeline for Blur and Scrambling: People vs. Vision Models.

Supports the 21-participant (3 Groups x 7 Slices: P01-P21) design:
- Group 1 (G1): P01..P07 (Slices 1..7, covering all 490 stimuli)
- Group 2 (G2): P08..P14 (Slices 1..7, covering all 490 stimuli)
- Group 3 (G3): P15..P21 (Slices 1..7, covering all 490 stimuli)

Tasks performed:
1. Optionally syncs completed participant submissions from the cloud registry into `human_logs/`.
2. Aggregates participant logs (`human_logs/responses_*.csv` and/or `VL1:...` tokens) into:
   - `human_results.csv` (global human accuracy + 2000-resample bootstrap 95% CIs)
   - `human_results_by_group.csv` (per-group accuracy + 2000-resample bootstrap 95% CIs for G1, G2, G3)
3. Evaluates CLIP ViT-L/14 and DINOv2 ViT-B/14 strictly on the 70 unique `human_uids`
   (`model_results_human_subset.csv`) alongside the full 1000-image test set (`model_results.csv`),
   producing `comparison_results.csv` and `comparison_figure.png`.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

BASE_SEED = 1234
N_BOOT = 2000
NTFY_TOPIC = "aims2026_visual_perception_v3_registry"

CATEGORIES = [
    "airplane", "bicycle", "car", "cat", "chair",
    "clock", "fish", "house", "tree", "umbrella",
]
BLUR_SIGMAS = {"low": 2, "medium": 5, "high": 10}
SHUFFLE_GRIDS = {"low": 2, "medium": 4, "high": 8}

CONDITIONS = [
    {"name": "intact", "family": "intact", "level": "none", "param": 0},
    {"name": "blur_low", "family": "blur", "level": "low", "param": BLUR_SIGMAS["low"]},
    {"name": "blur_medium", "family": "blur", "level": "medium", "param": BLUR_SIGMAS["medium"]},
    {"name": "blur_high", "family": "blur", "level": "high", "param": BLUR_SIGMAS["high"]},
    {"name": "shuffle_low", "family": "shuffle", "level": "low", "param": SHUFFLE_GRIDS["low"]},
    {"name": "shuffle_medium", "family": "shuffle", "level": "medium", "param": SHUFFLE_GRIDS["medium"]},
    {"name": "shuffle_high", "family": "shuffle", "level": "high", "param": SHUFFLE_GRIDS["high"]},
]
COND_NAMES = [c["name"] for c in CONDITIONS]
COND_META = {c["name"]: c for c in CONDITIONS}
MODEL_NAMES = ["CLIP ViT-L/14", "DINOv2 ViT-B/14"]
GROUPS = ["G1", "G2", "G3"]


def _build_slot_catalog(max_slots: int = 300) -> dict[str, tuple[str, int]]:
    """Build the deterministic PID -> (group, slice) catalog matching index.html (slots 1..max_slots)."""
    used: set[str] = set()
    catalog: dict[str, tuple[str, int]] = {}
    for slot_num in range(1, max_slots + 1):
        if slot_num <= 21:
            pid = f"P{slot_num:02d}"
            grp_idx = ((slot_num - 1) // 7) + 1
            slice_idx = ((slot_num - 1) % 7) + 1
        else:
            extra_idx = slot_num - 22
            grp_idx = (extra_idx % 3) + 1
            slice_idx = ((extra_idx // 3) % 7) + 1
            if extra_idx < 78:
                num = 22 + ((extra_idx * 17 + 12) % 78)
            else:
                num = 100 + (((extra_idx - 78) * 37 + 19) % 900)
            pid = f"P{num:02d}"
            while pid in used:
                num = 22 if num >= 999 else (num + 1)
                if num < 22:
                    num = 22
                pid = f"P{num:02d}"
        used.add(pid)
        catalog[pid] = (f"G{grp_idx}", slice_idx)
    return catalog


SLOT_CATALOG_BY_PID = _build_slot_catalog(300)


def infer_group_and_slice(pid: str) -> tuple[str, int]:
    """Map any allocated Participant ID (P01..P21 and extra allocated slots >21) to ('G1'..'G3', 1..7)."""
    cleaned = str(pid).strip().upper()
    if cleaned in SLOT_CATALOG_BY_PID:
        return SLOT_CATALOG_BY_PID[cleaned]
    m = re.search(r"(\d+)", cleaned)
    num = int(m.group(1)) if m else 1
    norm = f"P{num:02d}"
    if norm in SLOT_CATALOG_BY_PID:
        return SLOT_CATALOG_BY_PID[norm]
    base_idx = (num - 1) % 21
    grp_idx = (base_idx % 3) + 1
    slot_idx = ((base_idx // 3) % 7) + 1
    return f"G{grp_idx}", slot_idx


def bootstrap_ci(correct: np.ndarray, n_boot: int = N_BOOT, seed: int = BASE_SEED) -> tuple[float, float]:
    """Compute percentile 95% bootstrap CI (in percent) over binary correctness indicators."""
    arr = np.asarray(correct, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    accs = arr[idx].mean(axis=1) * 100.0
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


def decode_vl1_token(token_str: str) -> pd.DataFrame:
    """Decode a compact Base64 'VL1:...' result token into a DataFrame of trials."""
    cleaned = token_str.strip()
    if cleaned.startswith("VL1:"):
        cleaned = cleaned[4:]
    payload = json.loads(base64.b64decode(cleaned).decode("utf-8"))
    pid = str(payload.get("p", "P01")).strip()
    grp, slot = infer_group_and_slice(pid)
    rows = []
    for chunk in str(payload.get("d", "")).split(";"):
        if not chunk:
            continue
        trial, uid, c_idx, lbl_idx, rsp_idx, rt = [int(x) for x in chunk.split(",")]
        cond = COND_NAMES[c_idx]
        corr_lbl = CATEGORIES[lbl_idx]
        resp_lbl = CATEGORIES[rsp_idx]
        rows.append({
            "participant": pid,
            "group": str(payload.get("g", grp)),
            "slot_in_group": int(payload.get("s", slot)),
            "trial": trial,
            "uid": uid,
            "condition": cond,
            "file": f"s{uid:04d}_{cond}.png",
            "correct_label": corr_lbl,
            "response": resp_lbl,
            "correct": int(resp_lbl == corr_lbl),
            "rt_ms": rt,
        })
    return pd.DataFrame(rows)


def sync_cloud_submissions_to_logs(logs_dir: Path) -> int:
    """Fetch any cloud-recorded submissions from ntfy.sh and write `responses_PXX.csv` into `logs_dir`, respecting purge tombstones."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    try:
        url = f"https://ntfy.sh/{NTFY_TOPIC}/json?poll=1&since=all"
        resp = requests.get(url, timeout=8)
        if not resp.ok:
            return 0

        events: list[tuple[dict, int]] = []
        for line in resp.text.splitlines():
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
                if evt.get("event") != "message" or not evt.get("message"):
                    continue
                msg = json.loads(evt["message"])
                evt_time_ms = int(msg.get("tsMs") or (int(evt.get("time", 0)) * 1000))
                events.append((msg, evt_time_ms))
            except Exception:
                continue

        # Pass 1: Determine purge cutoff timestamps
        last_purge_ts = 0
        deleted_pids: dict[str, int] = {}
        for msg, ts_ms in events:
            mtype = msg.get("type", "")
            if mtype == "factory_purge" and ts_ms > last_purge_ts:
                last_purge_ts = ts_ms
            elif mtype == "delete_pid" and msg.get("participant"):
                pid = str(msg["participant"]).strip().upper()
                if ts_ms > deleted_pids.get(pid, 0):
                    deleted_pids[pid] = ts_ms

        # Pass 2: Write only valid submissions strictly newer than the cutoff
        valid_by_pid: dict[str, pd.DataFrame] = {}
        for msg, ts_ms in events:
            token = msg.get("token", "")
            if token.startswith("VL1:"):
                try:
                    df = decode_vl1_token(token)
                    if not df.empty:
                        pid = str(df["participant"].iloc[0]).strip().upper()
                        cutoff = max(last_purge_ts, deleted_pids.get(pid, 0))
                        if ts_ms > cutoff and pid not in valid_by_pid:
                            valid_by_pid[pid] = df
                except Exception:
                    continue

        for pid, df in valid_by_pid.items():
            out_file = logs_dir / f"responses_{pid}.csv"
            df.to_csv(out_file, index=False)
            saved_count += 1
    except Exception:
        return 0
    if saved_count > 0:
        print(f"[Cloud Sync] Pulled {saved_count} submission(s) into '{logs_dir}/'.")
    return saved_count


def aggregate_human_logs(
    logs_dir: Path,
    out_csv: Path,
    out_group_csv: Path,
    n_boot: int = N_BOOT,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Aggregate participant CSVs (P01..P21) in `logs_dir` into global `human_results.csv` and per-group CSV."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(logs_dir.glob("*.csv"))
    token_files = sorted(list(logs_dir.glob("*.txt")) + list(logs_dir.glob("*.b64")))
    if not csv_files and not token_files:
        print(f"[Human Logs] No participant CSV or token files found in '{logs_dir}/'.")
        return None, None, None

    frames = []
    required_cols = {"participant", "trial", "uid", "condition", "correct"}
    for path in csv_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        missing = required_cols - set(df.columns)
        if missing:
            print(f"[Human Logs] Skipping '{path.name}' (missing columns: {sorted(missing)}).")
            continue
        frames.append(df)

    for tpath in token_files:
        text = tpath.read_text(encoding="utf-8", errors="ignore")
        for match in re.findall(r"VL1:[A-Za-z0-9+/=_-]+", text):
            try:
                frames.append(decode_vl1_token(match))
            except Exception as exc:
                print(f"[Human Logs] Could not decode VL1 token in '{tpath.name}': {exc}")

    if not frames:
        print(f"[Human Logs] No valid participant logs found in '{logs_dir}/'.")
        return None, None, None

    raw = pd.concat(frames, ignore_index=True)
    raw["participant"] = raw["participant"].astype(str).str.strip()
    raw["condition"] = raw["condition"].astype(str).str.strip()
    raw["uid"] = pd.to_numeric(raw["uid"], errors="coerce").astype("Int64")
    raw["trial"] = pd.to_numeric(raw["trial"], errors="coerce").astype("Int64")
    raw["correct"] = pd.to_numeric(raw["correct"], errors="coerce")
    raw = raw.dropna(subset=["uid", "trial", "correct"])

    # Assign group (G1, G2, G3) and slot_in_group (1..7) from participant ID
    inferred = raw["participant"].map(infer_group_and_slice)
    raw["group"] = [g for g, _ in inferred]
    raw["slot_in_group"] = [s for _, s in inferred]

    before_dedup = len(raw)
    raw = raw.drop_duplicates(subset=["participant", "uid", "condition"], keep="last").reset_index(drop=True)
    if len(raw) < before_dedup:
        print(f"[Human Logs] Deduplicated {before_dedup - len(raw)} duplicate row(s).")

    participants = sorted(raw["participant"].unique())
    by_grp = {g: sorted(raw[raw["group"] == g]["participant"].unique()) for g in GROUPS}
    print(f"[Human Logs] Loaded {len(raw)} trials across {len(participants)} participant(s): {by_grp}")
    if "rt_ms" in raw.columns:
        rts = pd.to_numeric(raw["rt_ms"], errors="coerce").dropna()
        if not rts.empty:
            print(f"[Human Logs] Median RT: {rts.median():.0f} ms (IQR: {rts.quantile(0.25):.0f}-{rts.quantile(0.75):.0f} ms)")

    # 1. Global aggregation (human_results.csv)
    rows = []
    for cond in COND_NAMES:
        scores = raw.loc[raw["condition"] == cond, "correct"].to_numpy(dtype=float)
        if scores.size == 0:
            continue
        acc = float(scores.mean() * 100.0)
        lo, hi = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
        rows.append({
            "condition": cond,
            "accuracy": round(acc, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        })

    human_df = pd.DataFrame(rows, columns=["condition", "accuracy", "ci_low", "ci_high"])
    human_df.to_csv(out_csv, index=False)
    print(f"[Human Logs] Saved global human results -> '{out_csv}'")

    # 2. Per-group aggregation (G1, G2, G3 -> human_results_by_group.csv)
    group_rows = []
    for grp in GROUPS:
        gdf = raw[raw["group"] == grp]
        if gdf.empty:
            continue
        for cond in COND_NAMES:
            scores = gdf.loc[gdf["condition"] == cond, "correct"].to_numpy(dtype=float)
            if scores.size == 0:
                continue
            acc = float(scores.mean() * 100.0)
            lo, hi = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
            group_rows.append({
                "group": grp,
                "condition": cond,
                "n": int(scores.size),
                "n_correct": int(scores.sum()),
                "accuracy": round(acc, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
            })

    group_df = pd.DataFrame(group_rows) if group_rows else None
    if group_df is not None:
        group_df.to_csv(out_group_csv, index=False)
        print(f"[Human Logs] Saved per-group human results -> '{out_group_csv}'")

    return human_df, group_df, raw


def evaluate_models_on_human_subset(
    human_trials_csv: Path,
    model_predictions_csv: Path,
    model_results_csv: Path,
    out_subset_csv: Path,
    n_boot: int = N_BOOT,
    seed: int = BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Evaluate models strictly on the 70 unique `human_uids` alongside full N=1000 results."""
    trials = pd.read_csv(human_trials_csv)
    human_uids = sorted(trials["uid"].astype(int).unique().tolist())

    preds = pd.read_csv(model_predictions_csv)
    preds["uid"] = preds["uid"].astype(int)
    preds["correct"] = (preds["label"].astype(int) == preds["pred"].astype(int)).astype(float)

    sub_preds = preds[preds["uid"].isin(human_uids)].copy()

    subset_rows = []
    for model_name in MODEL_NAMES:
        for cond_name in COND_NAMES:
            meta = COND_META[cond_name]
            mask = (sub_preds["model"] == model_name) & (sub_preds["condition"] == cond_name)
            arr = sub_preds.loc[mask, "correct"].to_numpy(dtype=float)
            if arr.size == 0:
                raise ValueError(f"Missing predictions for model='{model_name}', condition='{cond_name}' on human_uids.")
            lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed)
            subset_rows.append({
                "model": model_name,
                "condition": cond_name,
                "family": meta["family"],
                "level": meta["level"],
                "param": meta["param"],
                "subset": "human_subset",
                "n": int(arr.size),
                "n_correct": int(arr.sum()),
                "accuracy": float(arr.mean() * 100.0),
                "ci_low": lo,
                "ci_high": hi,
            })

    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(out_subset_csv, index=False)
    full_df = pd.read_csv(model_results_csv)
    print(f"[Models] Evaluated {len(MODEL_NAMES)} models on {len(human_uids)} human_uids -> '{out_subset_csv}'")
    return full_df, subset_df, human_uids


def build_comparison_table(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    human_df: pd.DataFrame | None,
    group_df: pd.DataFrame | None,
    raw_human_df: pd.DataFrame | None,
    out_comparison_csv: Path,
) -> pd.DataFrame:
    """Create a unified comparison CSV aligning full N=1000, N=70 human_uids, per-group, and global human accuracy."""
    rows = []
    for cond in COND_NAMES:
        row: dict[str, object] = {"condition": cond}
        for model_name in MODEL_NAMES:
            slug = "clip" if model_name.startswith("CLIP") else "dinov2"
            f_rec = full_df[(full_df["model"] == model_name) & (full_df["condition"] == cond)].iloc[0]
            s_rec = subset_df[(subset_df["model"] == model_name) & (subset_df["condition"] == cond)].iloc[0]
            row[f"{slug}_full_n1000_acc"] = round(float(f_rec["accuracy"]), 2)
            row[f"{slug}_full_ci_low"] = round(float(f_rec["ci_low"]), 2)
            row[f"{slug}_full_ci_high"] = round(float(f_rec["ci_high"]), 2)
            row[f"{slug}_human70_acc"] = round(float(s_rec["accuracy"]), 2)
            row[f"{slug}_human70_ci_low"] = round(float(s_rec["ci_low"]), 2)
            row[f"{slug}_human70_ci_high"] = round(float(s_rec["ci_high"]), 2)
            row[f"{slug}_delta_70_vs_1000"] = round(float(s_rec["accuracy"]) - float(f_rec["accuracy"]), 2)

        if group_df is not None and not group_df.empty:
            for grp in GROUPS:
                g_match = group_df[(group_df["group"] == grp) & (group_df["condition"] == cond)]
                if not g_match.empty:
                    row[f"human_{grp}_acc"] = round(float(g_match.iloc[0]["accuracy"]), 2)

        if human_df is not None and not human_df.empty:
            h_match = human_df[human_df["condition"] == cond]
            if not h_match.empty:
                h_rec = h_match.iloc[0]
                row["human_acc"] = round(float(h_rec["accuracy"]), 2)
                row["human_ci_low"] = round(float(h_rec["ci_low"]), 2)
                row["human_ci_high"] = round(float(h_rec["ci_high"]), 2)
                if raw_human_df is not None:
                    row["human_n_trials"] = int((raw_human_df["condition"] == cond).sum())
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_comparison_csv, index=False)
    print(f"[Summary] Saved comparison table -> '{out_comparison_csv}'")
    return comp_df


def plot_comparison_figure(
    full_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    human_df: pd.DataFrame | None,
    out_path: Path,
    show_full_reference: bool = True,
) -> None:
    """Plot two-panel Blur | Tile Shuffle accuracy curve comparing models (full + 70 UIDs) and humans."""
    levels = ["low", "medium", "high"]
    xlab = {
        "blur": ["intact"] + [f"sigma={BLUR_SIGMAS[l]}" for l in levels],
        "shuffle": ["intact"] + [f"{SHUFFLE_GRIDS[l]**2} tiles" for l in levels],
    }
    style = {
        "CLIP ViT-L/14": ("tab:blue", "o"),
        "DINOv2 ViT-B/14": ("tab:green", "s"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, fam in zip(axes, ["blur", "shuffle"]):
        conds = ["intact"] + [f"{fam}_{l}" for l in levels]
        xs = np.arange(len(conds))

        for idx_m, (name, (col, mk)) in enumerate(style.items()):
            offset = -0.06 if idx_m == 0 else 0.06
            if show_full_reference:
                d_full = full_df[full_df["model"] == name].set_index("condition").loc[conds]
                ax.errorbar(
                    xs,
                    d_full["accuracy"],
                    yerr=[d_full["accuracy"] - d_full["ci_low"], d_full["ci_high"] - d_full["accuracy"]],
                    color=col,
                    marker=mk,
                    alpha=0.45,
                    ls="-",
                    capsize=3,
                    label=f"{name} (N=1000)",
                )
            d_sub = subset_df[subset_df["model"] == name].set_index("condition").loc[conds]
            ax.errorbar(
                xs + offset,
                d_sub["accuracy"],
                yerr=[d_sub["accuracy"] - d_sub["ci_low"], d_sub["ci_high"] - d_sub["accuracy"]],
                color=col,
                marker=mk,
                mfc="white",
                ls="--",
                capsize=3,
                label=f"{name} (70 human UIDs)",
            )

        if human_df is not None and not human_df.empty:
            h = human_df.set_index("condition").reindex(conds)
            valid = h["accuracy"].notna()
            if valid.any():
                ax.errorbar(
                    xs[valid],
                    h.loc[valid, "accuracy"],
                    yerr=[
                        h.loc[valid, "accuracy"] - h.loc[valid, "ci_low"],
                        h.loc[valid, "ci_high"] - h.loc[valid, "accuracy"],
                    ],
                    color="tab:red",
                    marker="^",
                    markersize=7,
                    linewidth=2.2,
                    ls="-",
                    capsize=3,
                    label="Humans",
                )

        ax.axhline(10, color="grey", ls=":", label="Chance (10%)")
        ax.set_xticks(xs)
        ax.set_xticklabels(xlab[fam])
        ax.set_ylim(0, 104)
        ax.set_title("Blur" if fam == "blur" else "Tile shuffle (7 px offset grid)")
        ax.set_xlabel("Damage level")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("10-way accuracy (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Figure] Saved figure -> '{out_path}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate 21-participant human logs and evaluate ViTs on 70 human_uids.")
    parser.add_argument("--human-logs", type=Path, default=Path("human_logs"), help="Directory of participant CSVs")
    parser.add_argument("--human-trials", type=Path, default=Path("human_trials.csv"), help="Input trial schedule CSV (P01-P21)")
    parser.add_argument("--model-results", type=Path, default=Path("model_results.csv"), help="Full N=1000 model results CSV")
    parser.add_argument("--model-predictions", type=Path, default=Path("model_predictions.csv"), help="Per-UID model predictions CSV")
    parser.add_argument("--out-human", type=Path, default=Path("human_results.csv"), help="Global aggregated human CSV output")
    parser.add_argument("--out-group-human", type=Path, default=Path("human_results_by_group.csv"), help="Per-group human CSV output")
    parser.add_argument("--out-subset", type=Path, default=Path("model_results_human_subset.csv"), help="70 human_uids model results CSV")
    parser.add_argument("--out-comparison", type=Path, default=Path("comparison_results.csv"), help="Side-by-side comparison CSV")
    parser.add_argument("--out-figure", type=Path, default=Path("comparison_figure.png"), help="Comparison figure PNG path")
    parser.add_argument("--sync-cloud", action="store_true", help="Pull cloud submissions into human_logs/ before aggregating")
    parser.add_argument("--update-main-figure", action="store_true", help="Always overwrite main_figure.png as well")
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="Bootstrap resamples (default: 2000)")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Random seed for bootstrap (default: 1234)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sync_cloud:
        sync_cloud_submissions_to_logs(args.human_logs)

    # (a) Aggregate participant logs (P01..P21) -> human_results.csv & human_results_by_group.csv
    human_df, group_df, raw_human_df = aggregate_human_logs(
        args.human_logs, args.out_human, args.out_group_human, n_boot=args.n_boot, seed=args.seed
    )
    if human_df is None and args.out_human.exists():
        human_df = pd.read_csv(args.out_human)
        print(f"[Human Logs] Loaded existing '{args.out_human}' for plotting.")

    # (b) Evaluate CLIP ViT-L/14 and DINOv2 ViT-B/14 on the 70 human_uids alongside N=1000 full test set
    full_df, subset_df, _ = evaluate_models_on_human_subset(
        args.human_trials,
        args.model_predictions,
        args.model_results,
        args.out_subset,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    comp_df = build_comparison_table(full_df, subset_df, human_df, group_df, raw_human_df, args.out_comparison)
    print("\n=== Model (N=1000 vs. 70 human_uids) & Human Summary (%) ===")
    print(comp_df.to_string(index=False))

    plot_comparison_figure(full_df, subset_df, human_df, args.out_figure)
    if human_df is not None or args.update_main_figure:
        plot_comparison_figure(full_df, subset_df, human_df, Path("main_figure.png"))


if __name__ == "__main__":
    main()
