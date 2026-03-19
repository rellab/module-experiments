import argparse
from pathlib import Path
import signal
import sys

import joblib
import pandas as pd
import numpy as np

# Default cache directory. Change this if needed.
DEFAULT_CACHE_DIR = Path("cache_offset_metraw")

signal.signal(signal.SIGPIPE, signal.SIG_DFL)

def get_consistent_target_modules(faultdata_by_t):
    """
    Use the same logic as training: keep only modules whose cumulative fault count
    is at least 1 for every `t_end`.
    """
    if not faultdata_by_t:
        return set()
    
    # Initialize with all modules present in the first step.
    target_modules = set(faultdata_by_t[0].keys())
    
    # Check observed data at every step and take the intersection.
    for fd in faultdata_by_t:
        # Keep only modules whose cumulative faults are greater than 0 at that step.
        current_step_active = {k for k, v in fd.items() if sum(v.fault) > 0}
        target_modules &= current_step_active
        
    return target_modules

def load_matching_faultdata(fault_files, baseline_files):
    """
    Load the best matching `faultdata` entry from the cache.
    """
    baseline_module_counts = []
    for baseline_file in baseline_files:
        data = joblib.load(baseline_file)
        baseline_module_counts.append(len(data["models"]))

    candidates = []
    for fault_file in fault_files:
        faultdata = joblib.load(fault_file)
        step_counts = [len(step) for step in faultdata]
        matched_steps = sum(count in baseline_module_counts for count in step_counts[:-1])
        candidates.append((matched_steps, max(step_counts), fault_file, faultdata))

    if not candidates:
        return None
    
    candidates.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    return candidates[0][3]

def build_baseline_map(baseline_files):
    baseline_map = {}
    for baseline_file in baseline_files:
        data = joblib.load(baseline_file)
        key = (float(data["train_t"]), float(data["test_t"]))
        baseline_map.setdefault(key, []).append(data)
    return baseline_map


def choose_baseline_for_step(baseline_candidates, fd0, fd1):
    if not baseline_candidates:
        return None

    target_set = set(fd0) & set(fd1)
    best = None
    best_score = None
    for candidate in baseline_candidates:
        model_set = set(candidate.get("models", {}).keys())
        mismatch = len(target_set - model_set) + len(model_set - target_set)
        score = (mismatch, abs(len(model_set) - len(target_set)))
        if best is None or score < best_score:
            best = candidate
            best_score = score

    return best

def extract_omega(model) -> float:
    # In many NHPP models, `params_[0]` corresponds to omega (a).
    try:
        return round(float(model.params_[0]), 4)
    except:
        return 0.0

def get_metrics_path(cache_dir: Path) -> Path:
    candidate = cache_dir.parent / "module_metrics.csv"
    if candidate.exists():
        return candidate
    return Path("module_metrics.csv")

def load_kloc_by_module(cache_dir: Path) -> pd.Series:
    path = get_metrics_path(cache_dir)
    if not path.exists():
        # Fall back to 1.0 temporarily if the metrics file is missing, for density calculation.
        return pd.Series()
    metrics = pd.read_csv(path, index_col=0)
    if "sum_nloc" not in metrics.columns:
        return pd.Series()
    kloc = metrics["sum_nloc"].astype(float) / 1000.0
    return kloc

def model_pred_err(models, fd0, fd1, target_modules):
    """
    Recompute the prediction error using only the shared target modules.
    """
    active_modules = set(models) & set(fd0) & set(fd1) & target_modules
    if not active_modules:
        return 0.0, 0

    err = 0.0
    for module_name in active_modules:
        t0 = fd0[module_name].total_time
        t1 = fd1[module_name].total_time
        predicted = models[module_name].mvf(t1) - models[module_name].mvf(t0)
        actual = sum(fd1[module_name].fault) - sum(fd0[module_name].fault)
        err += abs(predicted - actual)
    return float(err), len(active_modules)

def select_sweep_entry(sweep_data, model_kind, lambda_value):
    if model_kind == "best":
        candidates = [(str(k), v) for k, v in sweep_data.items() if "prederr" in v and "result" in v]
        if not candidates:
            return None
        candidates.sort(key=lambda item: float(item[1]["prederr"]))
        return candidates[0][0], candidates[0][1]

    if model_kind == "lambda":
        if lambda_value is None:
            return None
        exact_key = str(lambda_value)
        if exact_key in sweep_data:
            entry = sweep_data[exact_key]
            if "prederr" in entry and "result" in entry:
                return exact_key, entry
        for key, entry in sweep_data.items():
            try:
                if abs(float(key) - lambda_value) < 1e-12 and "prederr" in entry and "result" in entry:
                    return str(key), entry
            except ValueError:
                continue
    return None

def choose_lambda_result_for_step(fd0, fd1, sweep_list, model_kind, lambda_value, target_modules, baseline_models):
    best_exact = None
    best_fallback = None
    baseline_set = set(baseline_models)

    for sweep_file, sweep_data in sweep_list:
        selected = select_sweep_entry(sweep_data, model_kind, lambda_value)
        if selected is None:
            continue
        
        chosen_lambda, entry = selected
        models = entry.get("result", {}).get("models")
        if not isinstance(models, dict):
            continue

        measured_err, used_count = model_pred_err(models, fd0, fd1, target_modules)
        if used_count == 0:
            continue

        model_set = set(models)
        missing_vs_baseline = len(baseline_set - model_set)
        extra_vs_baseline = len(model_set - baseline_set)
        module_mismatch = missing_vs_baseline + extra_vs_baseline

        cached_prederr = float(entry["prederr"])
        prederr_gap = abs(measured_err - cached_prederr)
        measured_mae = measured_err / used_count
        
        item = {
            "file": sweep_file,
            "lambda_key": chosen_lambda,
            "entry": entry,
            "cached_prederr": cached_prederr,
            "measured_prederr": measured_err,
            "measured_mae": measured_mae,
            "used_count": used_count,
            "module_mismatch": module_mismatch,
            "prederr_gap": prederr_gap,
        }

        if item["module_mismatch"] == 0:
            if best_exact is None or item["prederr_gap"] < best_exact["prederr_gap"]:
                best_exact = item
            continue

        if best_fallback is None:
            best_fallback = item
            continue

        current_score = (item["module_mismatch"], item["prederr_gap"])
        fallback_score = (best_fallback["module_mismatch"], best_fallback["prederr_gap"])
        if current_score < fallback_score:
            best_fallback = item

    return best_exact if best_exact is not None else best_fallback

def summarize_step(fd0, fd1, models, kloc_by_module, target_modules, omega_by_module=None):
    """
    Summarize only the consistent shared module set (`target_modules`).
    """
    analysis = []
    for module_name in sorted(target_modules):
        if module_name not in models or module_name not in fd0 or module_name not in fd1:
            continue
            
        t0 = fd0[module_name].total_time
        t1 = fd1[module_name].total_time
        model = models[module_name]
        
        kloc = float(kloc_by_module.get(module_name, 1.0))
        # Use the estimated omega if available; otherwise use the model's first parameter.
        omega = round(float(omega_by_module[module_name]), 4) if (omega_by_module and module_name in omega_by_module) else extract_omega(model)
        
        train_total_fault = sum(fd0[module_name].fault)
        predicted = model.mvf(t1) - model.mvf(t0)
        actual = sum(fd1[module_name].fault) - sum(fd0[module_name].fault)
        
        analysis.append({
            "module": module_name,
            "kloc": round(kloc, 4),
            "omega": omega,
            "omega_density": round(omega / kloc, 4) if kloc > 0 else 0,
            "train_total_fault": train_total_fault,
            "predicted": round(predicted, 4),
            "actual": actual,
            "abs_error": round(abs(predicted - actual), 4),
            "diff": round(actual - predicted, 4),
        })
    return pd.DataFrame(analysis)

def build_parser(default_view="topbottom"):
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", nargs="?", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("limit", nargs="?", type=int, default=10)
    parser.add_argument("--view", choices=["topbottom", "all"], default=default_view)
    parser.add_argument("--model", choices=["baseline", "best", "lambda"], default="baseline")
    parser.add_argument("--lambda-value", type=float, default=None)
    parser.add_argument("--output", choices=["text", "csv"], default="text")
    return parser

def run(args):
    cache_dir = Path(args.cache_dir)
    limit = max(1, int(args.limit))
    fault_files = sorted(cache_dir.glob("faultdata*.joblib"))
    baseline_files = sorted(cache_dir.glob("baseline*.joblib"))
    sweep_files = sorted(cache_dir.glob("lambda_sweep*.joblib"))

    if not fault_files or not baseline_files:
        print("Error: No faultdata found.")
        return

    if args.model == "lambda" and args.lambda_value is None:
        raise ValueError("Specify --lambda-value when using --model lambda.")

    # 1. Load data.
    faultdata_by_t = load_matching_faultdata(fault_files, baseline_files)
    if faultdata_by_t is None:
        print("Error: No valid faultdata found.")
        return
    baseline_map = build_baseline_map(baseline_files)
    kloc_by_module = load_kloc_by_module(cache_dir)

    # 2. Important: identify the targets shared across the full period using the same logic as training.
    consistent_targets = get_consistent_target_modules(faultdata_by_t)
    
    sweep_list = []
    if args.model in {"best", "lambda"}:
        for f in sweep_files:
            sweep_list.append((f, joblib.load(f)))
        if not sweep_list:
            raise FileNotFoundError("lambda_sweep cache was not found.")

    csv_rows = []

    for i in range(len(faultdata_by_t) - 1):
        fd0, fd1 = faultdata_by_t[i], faultdata_by_t[i+1]
        train_t = float(next(iter(fd0.values())).total_time)
        test_t = float(next(iter(fd1.values())).total_time)

        if args.output == "text":
            print(f"\n{'='*60}\n Step: {train_t:g} -> {test_t:g}\n{'='*60}")

        baseline_candidates = baseline_map.get((train_t, test_t), [])
        step_baseline = choose_baseline_for_step(baseline_candidates, fd0, fd1)
        if not step_baseline:
            continue

        models = step_baseline["models"]
        omega_by_module = None
        selected_lambda = None
        step_mae = None
        
        if args.model in {"best", "lambda"}:
            selected = choose_lambda_result_for_step(
                fd0,
                fd1,
                sweep_list,
                args.model,
                args.lambda_value,
                consistent_targets,
                step_baseline["models"],
            )
            if selected:
                res = selected["entry"]["result"]
                models = res["models"]
                if "names" in res and "omega" in res:
                    omega_by_module = dict(zip(res["names"], res["omega"]))
                selected_lambda = selected["lambda_key"]
                step_mae = selected["measured_mae"]
                if args.output == "text":
                    print(
                        f"model: {args.model}, "
                        f"lambda: {selected_lambda}, "
                        f"step_mae: {step_mae:.4f}, "
                        f"prederr_gap: {selected['prederr_gap']:.6f}, "
                        f"module_mismatch: {selected['module_mismatch']}"
                    )
            else:
                if args.output == "text":
                    print("No matching lambda_sweep cache was found.")
                continue
        else:
            if args.output == "text":
                print("model: baseline")

        # 3. Build a summary using only shared modules.
        df_res = summarize_step(fd0, fd1, models, kloc_by_module, consistent_targets, omega_by_module)
        
        if df_res.empty:
            if args.output == "text":
                print("No modules matched.")
            continue

        # Compute baseline step_mae here.
        if args.model == "baseline" and step_mae is None and not df_res.empty:
            step_mae = round(float(df_res["abs_error"].mean()), 6)

        if args.output == "text":
            print(f"Evaluated modules: {len(df_res)} (Total consistent: {len(consistent_targets)})")
        
        if args.view == "all":
            all_rows = df_res.sort_values("predicted", ascending=False)
            if args.output == "text":
                print(all_rows.to_string(index=False))
            else:
                tmp = all_rows.copy()
                tmp.insert(0, "selection", "all")
                csv_rows.append((train_t, test_t, tmp, selected_lambda, step_mae))
        else:
            top_rows = df_res.sort_values("predicted", ascending=False).head(limit)
            bottom_rows = df_res.sort_values("predicted", ascending=True).head(limit)
            if args.output == "text":
                print(f"\n[Top {limit}]\n", top_rows.to_string(index=False))
                print(f"\n[Bottom {limit}]\n", bottom_rows.to_string(index=False))
            else:
                top_tmp = top_rows.copy()
                top_tmp.insert(0, "selection", "top")
                bottom_tmp = bottom_rows.copy()
                bottom_tmp.insert(0, "selection", "bottom")
                merged = pd.concat([top_tmp, bottom_tmp], axis=0)
                csv_rows.append((train_t, test_t, merged, selected_lambda, step_mae))

    if args.output == "csv":
        if not csv_rows:
            return
        frames = []
        for train_t, test_t, frame, selected_lambda, step_mae in csv_rows:
            tmp = frame.copy()
            tmp.insert(0, "test_t", test_t)
            tmp.insert(0, "train_t", train_t)
            tmp.insert(0, "model", args.model)
            tmp.insert(0, "lambda", selected_lambda)
            tmp.insert(0, "step_mae", step_mae)
            frames.append(tmp)
        out = pd.concat(frames, axis=0, ignore_index=True)
        out.to_csv(sys.stdout, index=False)


def run_cli(argv=None, default_view="topbottom"):
    parser = build_parser(default_view=default_view)
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    run_cli()