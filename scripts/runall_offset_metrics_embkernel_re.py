# %%
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import pysrat.nhpp as srm
from pysrat.data import NHPPData, SMetricsData
from pysrat.nhpp.regression import fit_pr_nhpp

from sklearn.metrics.pairwise import cosine_similarity


# %%
CACHE_DIR = Path("cache_offset_metrics_re_embedding_kernel")
CACHE_DIR.mkdir(exist_ok=True)


def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.joblib"


def make_cache_key(*parts) -> str:
    s = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def joblib_load_if_exists(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def joblib_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


# %%
def load_embedding_dataframe(jsonl_path: str) -> pd.DataFrame:
    jsonl_path = Path(jsonl_path)

    folders = []
    vectors = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            folders.append(obj["folder"])
            vectors.append(obj["embedding"])

    if not vectors:
        raise ValueError("No embeddings found in file.")

    dim = len(vectors[0])
    columns = [f"b{i+1}" for i in range(dim)]
    df = pd.DataFrame(vectors, index=folders, columns=columns)
    df.index.name = "folder"
    return df


def folder_to_hash(folder: str) -> str:
    return hashlib.sha1(folder.encode("utf-8")).hexdigest()


def load_nhpp_from_folder(folder: str, faultdata_dir: str = "faultdata"):
    hash_value = folder_to_hash(folder)
    csv_path = Path(faultdata_dir) / f"{hash_value}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fault data not found: {csv_path}")

    data = NHPPData.from_csv(str(csv_path), intervals="week", counts="counts")
    return data


def load_all_data(df: pd.DataFrame, faultdata_dir: str = "faultdata") -> dict:
    data_dict = {}
    for folder in df.index:
        try:
            data_dict[folder] = load_nhpp_from_folder(folder, faultdata_dir)
        except (FileNotFoundError, ValueError):
            continue
    return data_dict


# %%
def data_truncation(data: dict, t_end: int) -> dict:
    print(f"Truncating to t_end={t_end}")
    dat = {}
    for k, d in data.items():
        try:
            truncated = d.truncate(t_end=t_end)
        except Exception:
            continue
        if truncated.total_fault > 0:
            dat[k] = truncated
    return dat


def pred_err(models: dict, faultdata0: dict, faultdata1: dict) -> float:
    err = 0.0
    for k, m in models.items():
        t0 = faultdata0[k].total_time
        t1 = faultdata1[k].total_time
        pred = m.mvf(t1) - m.mvf(t0)
        actual = sum(faultdata1[k].fault) - sum(faultdata0[k].fault)
        err += abs(pred - actual)
    return err


def fit_baseline_models(faultdata_train: dict, n_phases: int = 20) -> dict:
    return {k: srm.CanonicalPhaseTypeNHPP(n_phases).fit(d) for k, d in faultdata_train.items()}


# %%
def identity_random_effect_df(module_names: list[str]) -> pd.DataFrame:
    n = len(module_names)
    I = np.eye(n, dtype=float)
    return pd.DataFrame(I, index=module_names, columns=[f"re_{i+1}" for i in range(n)])


def cosine_precision(df: pd.DataFrame, jitter: float = 1e-6) -> pd.DataFrame:
    """
    df: rows=modules, cols=embedding dimensions
    return: precision matrix = inv(cosine kernel + jitter * I)
    """
    K = cosine_similarity(df.values)
    K = K + jitter * np.eye(K.shape[0], dtype=float)
    K_inv = np.linalg.inv(K)
    return pd.DataFrame(K_inv, index=df.index, columns=df.index)


def block_diag_l2_for_metrics_and_re(
    metric_cols: list[str],
    re_col_names: list[str],
    re_precision: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build block L2 matrix:

        [ 0_(q_metrics x q_metrics)      0 ]
        [ 0                              K^{-1} ]

    rows/cols correspond to [metric_cols..., re_col_names...]
    """
    q_metrics = len(metric_cols)
    q_re = len(re_col_names)

    full_cols = metric_cols + re_col_names
    L = np.zeros((q_metrics + q_re, q_metrics + q_re), dtype=float)

    # fill RE block
    L[q_metrics:, q_metrics:] = re_precision.to_numpy()

    return pd.DataFrame(L, index=full_cols, columns=full_cols)


# %%
def test_for_lambd(
    lambd: float,
    faultdata0: dict,
    faultdata1: dict,
    smetdat: SMetricsData,
    offset: np.ndarray,
    penalty: np.ndarray,
    l2matrix: pd.DataFrame,
    alpha: float = 0.0,
):
    models = fit_baseline_models(faultdata0, n_phases=20)

    result = fit_pr_nhpp(
        models,
        smetdat,
        offset=offset,
        initialize=True,
        alpha=alpha,
        lambd=lambd,
        penalty=penalty,
        l2matrix=l2matrix,
    )

    err = pred_err(models, faultdata0, faultdata1)
    return result, err


# %%
# load / align
df = load_embedding_dataframe("group_embeddings.jsonl")
met = pd.read_csv("module_metrics.csv", index_col=0)

diff = met.index.symmetric_difference(df.index)
df = df.drop(diff, errors="ignore")
met = met.drop(diff, errors="ignore")
assert set(df.index) == set(met.index), "Mismatch in folders after alignment"

faultdata_full = load_all_data(df)


# %%
# choose modules that have faults for ALL t_end (intersection)
t_end = [440, 455, 470, 485, 500]

target_modules = set(faultdata_full.keys())
for t in t_end:
    truncated_data = data_truncation(faultdata_full, t_end=t)
    target_modules = target_modules & set(truncated_data.keys())

print("# of modules before nloc filtering:", len(target_modules))


# %%
# keep only target modules and positive sum_nloc
df = df.loc[df.index.isin(target_modules)].copy()
met = met.loc[met.index.isin(target_modules)].copy()

met = met[met["sum_nloc"] > 0].copy()
df = df.loc[met.index].copy()

target_modules = set(df.index)
faultdata_full = {k: v for k, v in faultdata_full.items() if k in target_modules}

print("# of modules after nloc filtering:", len(target_modules))


# %%
# Build truncated datasets aligned to each t_end
cache_key = make_cache_key("faultdata_by_t", sorted(target_modules), t_end)
p = cache_path(f"faultdata_by_t_{cache_key}")
faultdata_by_t = joblib_load_if_exists(p)

if faultdata_by_t is None:
    faultdata_by_t = []
    for t in t_end:
        faultdata_by_t.append({k: d.truncate(t_end=t) for k, d in faultdata_full.items()})
    joblib_dump(faultdata_by_t, p)


# %%
# sort all objects by common module order
module_names = sorted(target_modules)
df = df.loc[module_names]
met = met.loc[module_names]
faultdata_by_t = [{k: fd[k] for k in module_names} for fd in faultdata_by_t]


# %%
# offset: log(KLOC), KLOC = sum_nloc / 1000
kloc = met["sum_nloc"].astype(float) / 1000.0
if (kloc <= 0).any():
    bad = kloc[kloc <= 0]
    raise ValueError(f"Non-positive KLOC found:\n{bad}")

offset = np.log(kloc.to_numpy())

print("offset summary")
print(pd.Series(offset, index=module_names).describe())


# %%
# metric columns
metric_cols = [
    "files",
    "functions",
    "sum_ccn",
    "sum_token",
    "avg_nloc",
    "avg_ccn",
    "avg_token",
    "max_nloc",
    "max_ccn",
    "max_token",
    "avg_params",
    "max_params",
]

missing_cols = [c for c in metric_cols if c not in met.columns]
if missing_cols:
    raise ValueError(f"Missing metric columns: {missing_cols}")

metx = met[metric_cols].copy()

print("metric columns used:", metric_cols)
print("n metric columns:", len(metric_cols))


# %%
# random effect design (identity columns)
re_key = make_cache_key("identity_random_effect", module_names)
p = cache_path(f"df_re_{re_key}")
df_re = joblib_load_if_exists(p)

if df_re is None:
    df_re = identity_random_effect_df(module_names)
    joblib_dump(df_re, p)

re_col_names = list(df_re.columns)


# %%
# full design = metrics + random effects
design_key = make_cache_key(
    "metrics_plus_random_effect_embedding_kernel",
    module_names,
    metric_cols,
    metx.shape,
    df_re.shape,
)
p = cache_path(f"df_design_{design_key}")
df_design = joblib_load_if_exists(p)

if df_design is None:
    df_design = pd.concat([metx, df_re], axis=1)
    joblib_dump(df_design, p)

smetdat = SMetricsData.from_dataframe(df_design, use_index_as_name=True)

print("modules:", len(target_modules))
print("metrics dim:", metx.shape[1])
print("random effect dim:", df_re.shape[1])
print("total design dim:", df_design.shape[1])


# %%
# RE precision from embedding kernel
kernel_name = "cosine"
kernel_jitter = 1e-6

re_precision_key = make_cache_key(
    "re_precision",
    kernel_name,
    kernel_jitter,
    list(df.index),
    df.shape,
)
p = cache_path(f"df_re_precision_{re_precision_key}")
df_re_precision = joblib_load_if_exists(p)

if df_re_precision is None:
    df_re_precision = cosine_precision(df, jitter=kernel_jitter)
    df_re_precision = df_re_precision.loc[module_names, module_names]
    joblib_dump(df_re_precision, p)

print("embedding dim:", df.shape[1])
print("RE precision dim:", df_re_precision.shape)
print("kernel:", kernel_name)
print("kernel jitter:", kernel_jitter)


# %%
# full L2 matrix for [metrics, RE]
# metrics block = 0, RE block = embedding precision
l2_key = make_cache_key(
    "full_l2matrix_metrics_plus_re_kernel",
    metric_cols,
    re_col_names,
    kernel_name,
    kernel_jitter,
    list(df.index),
)
p = cache_path(f"df_l2matrix_{l2_key}")
df_l2matrix = joblib_load_if_exists(p)

if df_l2matrix is None:
    df_l2matrix = block_diag_l2_for_metrics_and_re(
        metric_cols=metric_cols,
        re_col_names=re_col_names,
        re_precision=df_re_precision,
    )
    joblib_dump(df_l2matrix, p)

print("full l2matrix dim:", df_l2matrix.shape)


# %%
# penalty vector
# metrics: free (0)
# RE: penalized (1)
q_metrics = metx.shape[1]
q_re = df_re.shape[1]
penalty = np.r_[np.zeros(q_metrics, dtype=float), np.ones(q_re, dtype=float)]

print("penalty shape:", penalty.shape)
print("n free coeffs:", int((penalty == 0).sum()))
print("n penalized coeffs:", int((penalty == 1).sum()))


# %%
# sanity check: ordering consistency
expected_cols = metric_cols + re_col_names
assert list(df_design.columns) == expected_cols, "Design column order mismatch"
assert list(df_l2matrix.index) == expected_cols, "L2 row order mismatch"
assert list(df_l2matrix.columns) == expected_cols, "L2 col order mismatch"
assert len(penalty) == len(expected_cols), "Penalty length mismatch"


# %%
# baseline per step
baseline_results = []
for i in range(len(t_end) - 1):
    train_t = t_end[i]
    test_t = t_end[i + 1]

    key = make_cache_key("baseline", train_t, test_t, sorted(target_modules))
    p = cache_path(f"baseline_{key}")

    cached_base = joblib_load_if_exists(p)
    if cached_base is None:
        print(f"Fitting baseline NHPP models for t_end={train_t} with {len(faultdata_by_t[i])} modules...")
        models = fit_baseline_models(faultdata_by_t[i], n_phases=20)
        mae = pred_err(models, faultdata_by_t[i], faultdata_by_t[i + 1])
        cached_base = {"train_t": train_t, "test_t": test_t, "mae": mae, "models": models}
        joblib_dump(cached_base, p)

    print(f"[baseline] {train_t}->{test_t} MAE:", cached_base["mae"])
    baseline_results.append(cached_base)


# %%
# lambda CV per step
lambda_grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 700.0]

allresults = []
for i in range(len(t_end) - 1):
    train_t = t_end[i]
    test_t = t_end[i + 1]

    step_key = make_cache_key(
        "lambda_sweep",
        train_t,
        test_t,
        lambda_grid,
        sorted(target_modules),
        "alpha=0.0",
        "offset=log(sum_nloc/1000)",
        "design=metrics_free+identity_RE_penalized",
        "RE_structure=embedding_kernel_precision",
        metric_cols,
        kernel_name,
        kernel_jitter,
    )
    p = cache_path(f"lambda_sweep_{step_key}")
    cvresults = joblib_load_if_exists(p)

    if cvresults is None:
        cvresults = {}
        for lambd in lambda_grid:
            result, err = test_for_lambd(
                lambd=lambd,
                faultdata0=faultdata_by_t[i],
                faultdata1=faultdata_by_t[i + 1],
                smetdat=smetdat,
                offset=offset,
                penalty=penalty,
                l2matrix=df_l2matrix,
                alpha=0.0,
            )
            cvresults[str(lambd)] = {"result": result, "prederr": err}
            print(f"[lambda] {train_t}->{test_t} lambda={lambd} prederr={err}")
        joblib_dump(cvresults, p)
    else:
        for lambd_str, res in cvresults.items():
            print(f"[lambda] {train_t}->{test_t} lambda={lambd_str} prederr={res['prederr']}")

    allresults.append({"train_t": train_t, "test_t": test_t, "cv": cvresults})


# %%
# pick best lambda per step
best = []
for step in allresults:
    items = [(float(l), v["prederr"]) for l, v in step["cv"].items()]
    items.sort(key=lambda x: x[1])
    best_lambda, best_err = items[0]
    best.append((step["train_t"], step["test_t"], best_lambda, best_err))

print("Best lambda per step:", best)


# %%
# summary table
best_df = pd.DataFrame(best, columns=["train_t", "test_t", "best_lambda", "best_prederr"])
print(best_df)