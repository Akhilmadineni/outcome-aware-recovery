#!/usr/bin/env python3
"""A5P1 - Outcome-aware recovery of cloud deployment workflows under ambiguous API responses.

Runs every experiment of the paper end to end:
  * preprocessing (winsorising outliers, median imputation + missing flags, engineered features)
  * proposed cost-sensitive CART recovery policy (cost-proportionate weights + leaf Bayes-risk decision
    over six actions incl. abort), hyper-parameters chosen by inner CV on recovery cost
  * baselines: retry-always, fail-fast, multinomial logistic regression (argmax)
  * extra reference: logistic regression with the same Bayes-risk decision
  * ablation (components and evidence groups), depth x leaf-size sensitivity, cost-ratio sensitivity
  * 5-fold stratified CV (mean +/- std), paired t-test and bootstrap CI vs. best baseline
  * calibration (ECE) by congestion quartile
Outputs: ../results/*.csv, ../results/metrics.json, ../figures/*.png (300 dpi) and ../results/*.jpg
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
SEED = 2026
np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "A5P1_ambiguous_api_calls.csv")
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
T0 = time.time()

# ----------------------------------------------------------------------------------------------
# Actions and recovery cost model (minutes-equivalent)
# ----------------------------------------------------------------------------------------------
ACTIONS = ["mark_complete_and_continue", "retry_with_idempotency_key", "deduplicate_and_continue",
           "compensate_and_retry", "wait_and_reconcile", "abort_and_rollback"]
SHORT = ["Mark", "Retry", "Dedup", "Comp", "Wait", "Abort"]
LABELS = ACTIONS[:5]            # label set of correct_recovery_action
A2OUT = {"mark_complete_and_continue": "applied", "retry_with_idempotency_key": "not_applied",
         "deduplicate_and_continue": "applied_duplicate", "compensate_and_retry": "partially_applied",
         "wait_and_reconcile": "partially_applied"}
P = dict(P_inc=60.0, P_dup=20.0, P_miss=45.0, P_dis=10.0, t_chk=1.0, abort_fix=15.0)


def cost_tensor(df, p=P):
    """Return (n,5,6) cost C[i, true_action, chosen_action] in minutes and (n,5,6,4) incident counts
    [inconsistent state, duplicate/orphan resource, missing resource, disruption of a healthy resource]."""
    d = np.clip(df["expected_duration_s"].to_numpy(), 1, None) / 60.0
    noidem = (df["api_supports_idempotency"].to_numpy() == 0).astype(float)
    n = len(d)
    t_retry = d + 0.5
    t_comp = 0.5 * d + 1.0
    t_wait = np.maximum(2.0, d)
    t_chk = p["t_chk"] + 0 * d
    t_abort = p["abort_fix"] + 2.0 * d
    C = np.zeros((n, 5, 6)); I = np.zeros((n, 5, 6, 4))
    # true = mark (applied)
    C[:, 0] = np.stack([0 * d, t_retry + p["P_dup"] * noidem, t_chk + 0 * d, t_comp + t_retry + p["P_dis"],
                        t_wait + t_chk, t_abort], 1)
    I[:, 0, 1, 1] = noidem; I[:, 0, 3, 3] = 1
    # true = retry (not applied)
    C[:, 1] = np.stack([p["P_miss"] + 0 * d, t_retry, t_chk + p["P_miss"], 1.0 + t_retry,
                        t_wait + t_chk + t_retry, t_abort], 1)
    I[:, 1, 0, 2] = 1; I[:, 1, 2, 2] = 1
    # true = dedup (applied duplicate)
    C[:, 2] = np.stack([p["P_dup"] + 0 * d, t_retry + p["P_dup"] * (1 + noidem), t_chk + 0 * d,
                        t_comp + t_retry + p["P_dis"], t_wait + 2 * t_chk, t_abort], 1)
    I[:, 2, 0, 1] = 1; I[:, 2, 1, 1] = 1 + noidem; I[:, 2, 3, 3] = 1
    # true = compensate (partial, short op)
    C[:, 3] = np.stack([p["P_inc"] + 0 * d, t_retry + p["P_inc"], t_chk + p["P_inc"], t_comp + t_retry,
                        t_wait + t_chk + t_comp + t_retry, t_abort], 1)
    I[:, 3, 0, 0] = 1; I[:, 3, 1, 0] = 1; I[:, 3, 2, 0] = 1
    # true = wait (partial, long op still converging)
    C[:, 4] = np.stack([p["P_inc"] + 0 * d, t_retry + p["P_inc"], t_chk + p["P_inc"], t_comp + t_retry + d + p["P_dis"],
                        t_wait + t_chk, t_abort], 1)
    I[:, 4, 0, 0] = 1; I[:, 4, 1, 0] = 1; I[:, 4, 2, 0] = 1; I[:, 4, 3, 3] = 1
    return C, I


# ----------------------------------------------------------------------------------------------
# Data and preprocessing
# ----------------------------------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["timestamp"])
y = df["correct_recovery_action"].map({a: i for i, a in enumerate(LABELS)}).to_numpy()
y_out = df["true_outcome"].to_numpy()
COST, INC = cost_tensor(df)
n = len(df)
C_true = COST[np.arange(n), y]                 # (n,6) cost row of the true state

CAT = ["service", "cloud_provider", "region", "operation", "response_class", "async_operation_status"]
NUM = ["operation_is_async", "api_supports_idempotency", "idempotency_key_sent", "client_timeout_s", "elapsed_s",
       "client_retries", "payload_kb", "control_plane_congestion", "expected_duration_s", "describe_poll_delay_s",
       "describe_shows_resource", "audit_log_event_seen", "audit_log_lag_s", "matching_resource_tags"]
WINS = ["elapsed_s", "payload_kb", "audit_log_lag_s", "expected_duration_s"]
MISS = ["payload_kb", "control_plane_congestion", "audit_log_lag_s"]
GROUPS = {"describe": ["describe_poll_delay_s", "describe_shows_resource", "describe_issued"],
          "async": ["async_operation_status"],
          "audit": ["audit_log_event_seen", "audit_log_lag_s"],
          "tags": ["matching_resource_tags"]}


class Prep:
    """Fit on training fold only: winsorise, impute medians, add flags/engineered features, one-hot."""
    def __init__(self, drop=()):
        self.drop = set(drop)

    def fit(self, d):
        self.lo = {c: d[c].quantile(0.005) for c in WINS}
        self.hi = {c: d[c].quantile(0.995) for c in WINS}
        self.med = {c: d[c].median() for c in NUM}
        self.cats = {c: sorted(d[c].unique()) for c in CAT}
        return self

    def transform(self, d):
        X = pd.DataFrame(index=d.index)
        for c in NUM:
            v = d[c].astype(float)
            if c in MISS:
                X[c + "_missing"] = v.isna().astype(float)
            v = v.fillna(self.med[c])
            if c in WINS:
                v = v.clip(self.lo[c], self.hi[c])
            X[c] = v
        X["elapsed_over_timeout"] = X["elapsed_s"] / d["client_timeout_s"]
        X["elapsed_over_expected"] = X["elapsed_s"] / X["expected_duration_s"].clip(lower=1)
        X["describe_issued"] = (d["describe_shows_resource"] >= 0).astype(float)
        X["effective_idempotency"] = d["api_supports_idempotency"] * d["idempotency_key_sent"]
        X["hour"] = d["timestamp"].dt.hour
        for c in CAT:
            for v in self.cats[c]:
                X[f"{c}={v}"] = (d[c] == v).astype(float)
        keep = [c for c in X.columns if not any(c == g or c.startswith(g + "=") or c.startswith(g + "_")
                                                for g in self.drop)]
        return X[keep]


def drop_cols(groups):
    cols = []
    for g in groups:
        cols += GROUPS[g]
    return cols


# ----------------------------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------------------------
def cost_weights(Ctr, ytr, gamma=1.0):
    """Cost-proportionate example weight: (mean regret of the wrong label actions)^gamma, normalised to mean 1."""
    rows = Ctr[np.arange(len(ytr)), ytr][:, :5]
    best = rows[np.arange(len(ytr)), ytr]
    reg = ((rows.sum(1) - 5 * best) / 4.0) ** gamma
    return reg / reg.mean()


def leaf_posterior(tree, Xtr, ytr, Xte, alpha=1.0):
    """Unweighted Laplace-smoothed class frequencies in each leaf (removes the training re-weighting)."""
    ltr = tree.apply(Xtr); lte = tree.apply(Xte)
    leaves = np.unique(ltr)
    idx = {l: i for i, l in enumerate(leaves)}
    cnt = np.zeros((len(leaves), 5))
    np.add.at(cnt, (np.vectorize(idx.get)(ltr), ytr), 1)
    post = (cnt + alpha) / (cnt.sum(1, keepdims=True) + 5 * alpha)
    return post[np.vectorize(idx.get)(lte)]


def bayes_action(prob, Cte, allow_abort=True):
    """Minimum expected cost action: a* = argmin_a sum_k p_k C[k,a]."""
    R = np.einsum("nk,nka->na", prob, Cte)
    if not allow_abort:
        R[:, 5] = np.inf
    return R.argmin(1)


def fit_cart(Xtr, ytr, Ctr, depth, leaf, gamma=1.0, seed=SEED):
    w = cost_weights(Ctr, ytr, gamma) if gamma > 0 else None
    t = DecisionTreeClassifier(criterion="gini", max_depth=depth, min_samples_leaf=leaf, random_state=seed)
    t.fit(Xtr.values, ytr, sample_weight=w)
    return t


DEPTHS = [4, 6, 8, 10, 12, 14]
LEAVES = [10, 20, 40, 80]
GAMMAS = [0.0, 0.5, 1.0]


def realised(choice, idx):
    c = COST[idx, y[idx], choice]
    inc = INC[idx, y[idx], choice]
    return c, inc


def select_hparams(dtr, ytr, Ctr, drop=(), weighted=True, bayes=True):
    """Inner 3-fold CV on mean realised recovery cost over (depth, min leaf, weight exponent gamma)."""
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED + 1)
    gams = GAMMAS if weighted else [0.0]
    grid = np.zeros((len(DEPTHS), len(LEAVES), len(gams)))
    for a, b in skf.split(dtr, ytr):
        pr = Prep(drop).fit(dtr.iloc[a]); Xa = pr.transform(dtr.iloc[a]); Xb = pr.transform(dtr.iloc[b])
        for i, dpt in enumerate(DEPTHS):
            for j, lf in enumerate(LEAVES):
                for k, g in enumerate(gams):
                    t = fit_cart(Xa, ytr[a], Ctr[a], dpt, lf, g)
                    if bayes:
                        act = bayes_action(leaf_posterior(t, Xa.values, ytr[a], Xb.values), Ctr[b])
                    else:
                        act = t.predict(Xb.values)
                    grid[i, j, k] += Ctr[b][np.arange(len(b)), ytr[b], act].mean() / 3
    i, j, k = np.unravel_index(grid.argmin(), grid.shape)
    return DEPTHS[i], LEAVES[j], gams[k], grid


def run_cart(dtr, dte, tr, te, drop=(), weighted=True, bayes=True, hp=None):
    ytr = y[tr]; Ctr = COST[tr]; Cte = COST[te]
    grid = None
    if hp is None:
        dpt, lf, gm, grid = select_hparams(dtr, ytr, Ctr, drop, weighted, bayes)
    else:
        dpt, lf, gm = hp
    pr = Prep(drop).fit(dtr); Xtr = pr.transform(dtr); Xte = pr.transform(dte)
    t = fit_cart(Xtr, ytr, Ctr, dpt, lf, gm)
    prob = leaf_posterior(t, Xtr.values, ytr, Xte.values)
    act = bayes_action(prob, Cte) if bayes else t.predict(Xte.values)
    return act, prob, dict(depth=dpt, leaf=lf, gamma=gm, leaves=int(t.get_n_leaves()), grid=grid, tree=t,
                           cols=list(Xtr.columns))


def run_lr(dtr, dte, tr, te, bayes=False):
    pr = Prep().fit(dtr); Xtr = pr.transform(dtr); Xte = pr.transform(dte)
    sc = StandardScaler().fit(Xtr.values)
    m = LogisticRegression(C=1.0, max_iter=3000, random_state=SEED)
    m.fit(sc.transform(Xtr.values), y[tr])
    prob = m.predict_proba(sc.transform(Xte.values))
    act = bayes_action(prob, COST[te]) if bayes else prob.argmax(1)
    return act, prob


def ece(prob, ytrue, bins=10):
    conf = prob.max(1); pred = prob.argmax(1); ok = (pred == ytrue).astype(float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for k in range(bins):
        m = (conf > edges[k]) & (conf <= edges[k + 1])
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def summarize(name, fold, te, act, prob=None):
    c, inc = realised(act, te)
    opt = COST[te, y[te], y[te]]
    lab = np.where(act == 5, -1, act)
    pred_out = np.array([A2OUT.get(LABELS[a], "abort") if a < 5 else "abort" for a in act])
    return dict(method=name, fold=fold,
                cost_min=c.mean(), regret_min=(c - opt).mean(),
                inconsistent_per1k=1000 * inc[:, 0].mean(), dup_orphan_per1k=1000 * inc[:, 1].mean(),
                missing_per1k=1000 * inc[:, 2].mean(), disrupt_per1k=1000 * inc[:, 3].mean(),
                action_acc=accuracy_score(y[te], lab), action_macro_f1=f1_score(y[te], lab, labels=range(5),
                                                                                   average="macro"),
                outcome_macro_f1=f1_score(y_out[te], pred_out, labels=sorted(set(y_out)), average="macro"),
                abort_rate=(act == 5).mean(), wait_rate=(act == 4).mean(),
                map_action_macro_f1=np.nan if prob is None else f1_score(y[te], prob.argmax(1), average="macro"),
                map_outcome_macro_f1=np.nan if prob is None else f1_score(
                    y_out[te], np.array([A2OUT[LABELS[k]] for k in prob.argmax(1)]), average="macro"))


# ----------------------------------------------------------------------------------------------
# Main 5-fold CV
# ----------------------------------------------------------------------------------------------
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, abl, per_call, cal_rows, hp_rows, grids, cms, imps = [], [], [], [], [], [], [], []
ABL = [("Full proposed (weights + Bayes leaf rule + abort)", dict()),
       ("No cost weights (gamma=0, Bayes rule)", dict(weighted=False)),
       ("No Bayes rule (CV-weighted CART, argmax)", dict(bayes=False)),
       ("Plain CART (no weights, argmax)", dict(weighted=False, bayes=False)),
       ("- describe evidence", dict(drop=("describe_poll_delay_s", "describe_shows_resource", "describe_issued"))),
       ("- async status", dict(drop=("async_operation_status",))),
       ("- audit log", dict(drop=("audit_log_event_seen", "audit_log_lag_s"))),
       ("- resource tags", dict(drop=("matching_resource_tags",))),
       ("Response + context only (no evidence)", dict(drop=tuple(drop_cols(GROUPS)) + ("async_operation_status",)))]
cong = df["control_plane_congestion"].to_numpy()
qedges = np.nanquantile(cong, [0.25, 0.5, 0.75])
cq = np.where(np.isnan(cong), -1, np.digitize(cong, qedges))

for f, (tr, te) in enumerate(skf.split(df, y)):
    dtr, dte = df.iloc[tr], df.iloc[te]
    # proposed
    act_p, prob_p, info = run_cart(dtr, dte, tr, te)
    grids.append(info["grid"]); hp_rows.append(dict(fold=f, depth=info["depth"], min_leaf=info["leaf"], gamma=info["gamma"],
                                                     leaves=info["leaves"]))
    imps.append(pd.Series(info["tree"].feature_importances_, index=info["cols"]))
    cms.append(confusion_matrix(y[te], act_p, labels=range(6))[:5])
    # baselines
    act_ra = np.full(len(te), 1)                       # retry-always
    act_ff = np.full(len(te), 5)                       # fail-fast: abort + rollback
    act_wr = np.full(len(te), 4)                       # always-reconcile (poll) reference
    act_lr, prob_lr = run_lr(dtr, dte, tr, te)
    act_lrb, _ = run_lr(dtr, dte, tr, te, bayes=True)
    for name, act, pb in [("Retry-always", act_ra, None), ("Fail-fast", act_ff, None),
                          ("Always-reconcile", act_wr, None), ("Logistic regression", act_lr, prob_lr),
                          ("LR + Bayes-risk rule", act_lrb, prob_lr), ("Proposed CS-CART", act_p, prob_p)]:
        rows.append(summarize(name, f, te, act, pb))
        c, _ = realised(act, te)
        per_call.append(pd.DataFrame(dict(method=name, fold=f, idx=te, cost=c)))
        for q in range(4):
            m = cq[te] == q
            per_call_q = c[m].mean()
            cal_rows.append(dict(method=name, fold=f, quartile=q + 1, cost_min=per_call_q))
    for q in range(4):
        m = cq[te] == q
        cal_rows.append(dict(method="ECE CS-CART", fold=f, quartile=q + 1, cost_min=ece(prob_p[m], y[te][m])))
        cal_rows.append(dict(method="ECE LR", fold=f, quartile=q + 1, cost_min=ece(prob_lr[m], y[te][m])))
    # ablation
    for name, kw in ABL:
        if name.startswith("Full"):
            act, pb = act_p, prob_p
        else:
            if "drop" in kw:   # evidence ablations reuse the fold's selected hyper-parameters
                kw = dict(kw, hp=(info["depth"], info["leaf"], info["gamma"]))
            act, pb, _ = run_cart(dtr, dte, tr, te, **kw)
        r = summarize(name, f, te, act, pb)
        r["describe_latency_s"] = 0.0 if "describe" in str(kw.get("drop", ())) else \
            dte["describe_poll_delay_s"].mean()
        abl.append(r)
    print(f"fold {f} done {time.time() - T0:.0f}s", flush=True)

res = pd.DataFrame(rows); res.to_csv(os.path.join(RES, "main_comparison_per_fold.csv"), index=False)
ab = pd.DataFrame(abl); ab.to_csv(os.path.join(RES, "ablation_per_fold.csv"), index=False)
pd.DataFrame(hp_rows).to_csv(os.path.join(RES, "selected_hyperparameters.csv"), index=False)
calq = pd.DataFrame(cal_rows); calq.to_csv(os.path.join(RES, "congestion_quartile_per_fold.csv"), index=False)
METRICS = ["cost_min", "regret_min", "inconsistent_per1k", "dup_orphan_per1k", "missing_per1k", "disrupt_per1k",
           "action_acc", "action_macro_f1", "outcome_macro_f1", "abort_rate", "wait_rate",
           "map_action_macro_f1", "map_outcome_macro_f1"]
ORDER = ["Retry-always", "Fail-fast", "Always-reconcile", "Logistic regression", "LR + Bayes-risk rule",
         "Proposed CS-CART"]


def ms(g, cols):
    out = {}
    for c in cols:
        out[c + "_mean"] = g[c].mean(); out[c + "_std"] = g[c].std(ddof=1)
    return pd.Series(out)


main = res.groupby("method").apply(lambda g: ms(g, METRICS)).loc[ORDER]
main.to_csv(os.path.join(RES, "table_main_comparison.csv"))
abl_t = ab.groupby("method", sort=False).apply(lambda g: ms(g, METRICS + ["describe_latency_s"]))
abl_t.to_csv(os.path.join(RES, "table_ablation.csv"))

# per-class recall of the proposed policy (pooled over folds)
cm = np.sum(cms, 0)
pd.DataFrame(cm, index=SHORT[:5], columns=SHORT).to_csv(os.path.join(RES, "confusion_matrix_proposed.csv"))
recall = {SHORT[k]: cm[k, k] / cm[k].sum() for k in range(5)}

# significance: proposed vs best fixed baseline (by cost) and vs LR
pc = pd.concat(per_call)
base_names = ["Retry-always", "Fail-fast", "Logistic regression"]
best_base = main.loc[base_names, "cost_min_mean"].idxmin()
sig = []
rng = np.random.default_rng(SEED)
for b in base_names + ["Always-reconcile", "LR + Bayes-risk rule"]:
    a1 = res[res.method == "Proposed CS-CART"].sort_values("fold")["cost_min"].to_numpy()
    b1 = res[res.method == b].sort_values("fold")["cost_min"].to_numpy()
    t, p = stats.ttest_rel(a1, b1)
    ca = pc[pc.method == "Proposed CS-CART"].sort_values("idx")["cost"].to_numpy()
    cb = pc[pc.method == b].sort_values("idx")["cost"].to_numpy()
    diff = ca - cb
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    w = stats.wilcoxon(ca, cb)
    sig.append(dict(comparison=f"Proposed vs {b}", mean_diff_min=diff.mean(), ci_low=np.percentile(boots, 2.5),
                    ci_high=np.percentile(boots, 97.5), paired_t_folds=t, p_t_folds=p, p_wilcoxon_calls=w.pvalue))
sig = pd.DataFrame(sig); sig.to_csv(os.path.join(RES, "table_significance.csv"), index=False)

# depth x leaf sensitivity (inner-CV cost averaged over folds)
G3 = np.mean(grids, 0)
G = G3.min(2)
gamma_curve = {str(g): float(G3[:, :, k].min()) for k, g in enumerate(GAMMAS)}
pd.DataFrame(G, index=[f"depth={d}" for d in DEPTHS], columns=[f"leaf={l}" for l in LEAVES]).to_csv(
    os.path.join(RES, "sensitivity_depth_leaf.csv"))

# cost-ratio sensitivity: scale the inconsistency penalty P_inc (retrain + evaluate under same scale), fold 0..4
sens = []
for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
    p2 = dict(P); p2["P_inc"] = P["P_inc"] * scale; p2["P_miss"] = P["P_miss"] * scale
    C2, I2 = cost_tensor(df, p2)
    for f, (tr, te) in enumerate(skf.split(df, y)):
        if f >= 3:
            break
        dtr, dte = df.iloc[tr], df.iloc[te]
        pr = Prep().fit(dtr); Xtr = pr.transform(dtr); Xte = pr.transform(dte)
        hp = hp_rows[f]
        t = fit_cart(Xtr, y[tr], C2[tr], hp["depth"], hp["min_leaf"], hp["gamma"])
        act = bayes_action(leaf_posterior(t, Xtr.values, y[tr], Xte.values), C2[te])
        inc = I2[te, y[te], act]
        sens.append(dict(scale=scale, fold=f, cost_min=C2[te, y[te], act].mean(),
                         inconsistent_per1k=1000 * inc[:, 0].mean(), wait_comp_share=np.isin(act, [3, 4]).mean(),
                         abort_rate=(act == 5).mean(), retry_always_cost=C2[te, y[te], 1].mean()))
sens = pd.DataFrame(sens); sens.to_csv(os.path.join(RES, "sensitivity_penalty_per_fold.csv"), index=False)
sens_t = sens.groupby("scale").agg(["mean", "std"]); sens_t.to_csv(os.path.join(RES, "table_sensitivity_penalty.csv"))

# feature importance (mean over folds, top 12)
imp = pd.concat(imps, axis=1).fillna(0).mean(1).sort_values(ascending=False)
imp.to_csv(os.path.join(RES, "feature_importance.csv"), header=["gini_importance"])

# dataset summary table
ds = pd.DataFrame([
    ("Calls (rows) / columns", f"{len(df):,} / {df.shape[1]}"),
    ("Period", f"{df.timestamp.min():%Y-%m-%d} to {df.timestamp.max():%Y-%m-%d}"),
    ("Services / providers / regions / operations",
     f"{df.service.nunique()} / {df.cloud_provider.nunique()} / {df.region.nunique()} / {df.operation.nunique()}"),
    ("Response classes", f"{df.response_class.nunique()} (timeout {100 * (df.response_class == 'timeout').mean():.1f}%)"),
    ("Async ops / idempotency key sent", f"{100 * df.operation_is_async.mean():.1f}% / "
                                         f"{100 * df.idempotency_key_sent.mean():.1f}%"),
    ("Describe issued / audit event seen", f"{100 * (df.describe_shows_resource >= 0).mean():.1f}% / "
                                           f"{100 * df.audit_log_event_seen.mean():.1f}%"),
    ("Missing (payload / congestion / audit lag)",
     " / ".join(f"{100 * df[c].isna().mean():.1f}%" for c in ["payload_kb", "control_plane_congestion",
                                                               "audit_log_lag_s"])),
    ("Outcome: applied / not / partial / duplicate",
     " / ".join(f"{100 * (df.true_outcome == o).mean():.1f}%" for o in
                ["applied", "not_applied", "partially_applied", "applied_duplicate"])),
    ("Action: Mark / Retry / Dedup / Comp / Wait",
     " / ".join(f"{100 * (df.correct_recovery_action == a).mean():.1f}%" for a in LABELS)),
], columns=["Property", "Value"])
ds.to_csv(os.path.join(RES, "table_dataset_summary.csv"), index=False)
oracle_cost = COST[np.arange(n), y, y].mean()

# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})
GREY = "#9a9a9a"; ACC = "#c0392b"; LRC = "#5d7ea8"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RES, name + ".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 92})
    plt.close(fig)


# Fig 1: methodology flowchart (Graphviz)
import sys
sys.path.insert(0, HERE)
import flowchart
flowchart.make(FIG, RES)

# Fig 2: cost and incidents per method (horizontal bars, shared method axis)
labs = ["Retry-always", "Fail-fast", "Always-reconcile", "Logistic regr.", "LR + Bayes rule", "CS-CART (prop.)"]
cols = [GREY, GREY, GREY, LRC, LRC, ACC]
fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.9), sharey=True)
x = np.arange(6)[::-1]
ax[0].barh(x, main["cost_min_mean"], xerr=main["cost_min_std"], color=cols, capsize=2, height=0.7)
ax[0].axvline(oracle_cost, ls=":", c="k", lw=0.8)
ax[0].text(oracle_cost + 0.4, -0.75, "oracle", fontsize=6.3, va="center"); ax[0].set_ylim(-1.1, 5.5)
ax[0].set_xlabel("Cost (min/call)")
allinc = main["inconsistent_per1k_mean"] + main["dup_orphan_per1k_mean"] + main["missing_per1k_mean"] \
    + main["disrupt_per1k_mean"]
ax[1].barh(x, allinc, color=cols, height=0.7, alpha=0.35, label="all incidents")
ax[1].barh(x, main["inconsistent_per1k_mean"], xerr=main["inconsistent_per1k_std"], color=cols, capsize=2,
           height=0.7, label="inconsistent")
ax[1].set_xlabel("Incidents / 1000 calls")
ax[1].legend(frameon=False, loc="lower right", fontsize=6.3)
ax[0].set_yticks(x); ax[0].set_yticklabels(labs, fontsize=7)
fig.tight_layout(w_pad=0.4)
save(fig, "fig2_policy_comparison")

# Fig 3: confusion matrix (row-normalised)
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(3.0, 2.25))
im = ax.imshow(cmn, cmap="Reds", vmin=0, vmax=1)
for i in range(5):
    for j in range(6):
        ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                color="white" if cmn[i, j] > 0.6 else "black")
ax.set_xticks(range(6)); ax.set_xticklabels(SHORT); ax.set_yticks(range(5)); ax.set_yticklabels(SHORT[:5])
ax.set_xlabel("Action chosen by CS-CART"); ax.set_ylabel("Correct action")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Row share")
save(fig, "fig3_confusion")

# Fig 4: depth x leaf sensitivity heatmap
fig, ax = plt.subplots(figsize=(3.0, 2.1))
im = ax.imshow(G, cmap="viridis_r", aspect="auto")
for i in range(len(DEPTHS)):
    for j in range(len(LEAVES)):
        ax.text(j, i, f"{G[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                color="white" if G[i, j] > G.min() + 0.6 * (G.max() - G.min()) else "black")
ax.set_xticks(range(len(LEAVES))); ax.set_xticklabels(LEAVES); ax.set_yticks(range(len(DEPTHS)))
ax.set_yticklabels(DEPTHS)
ax.set_xlabel("Minimum samples per leaf $n_{min}$"); ax.set_ylabel("Maximum depth $D$")
fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label="Inner-CV cost (min/call)")
save(fig, "fig4_depth_leaf_sensitivity")

# Fig 5: feature importance top 10
top = imp.head(10)[::-1]
UNIT = {"expected_duration_s": "expected duration (s)", "payload_kb": "payload (KB)",
        "audit_log_lag_s": "audit log lag (s)", "elapsed_s": "elapsed time (s)",
        "describe_poll_delay_s": "describe poll delay (s)", "client_timeout_s": "client timeout (s)"}
pretty = {k: UNIT.get(k, k.replace("_", " ").replace("=", ": ")) for k in top.index}
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.barh(range(len(top)), top.values, color=ACC)
ax.set_yticks(range(len(top))); ax.set_yticklabels([pretty[k] for k in top.index], fontsize=6.8)
ax.set_xlabel("Mean Gini importance (5 folds)")
save(fig, "fig5_feature_importance")

# Fig 6: cost by congestion quartile
cq_t = calq.groupby(["method", "quartile"])["cost_min"].agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(3.3, 2.0))
for k, (m, c, mk) in enumerate([("Always-reconcile", GREY, "s"), ("Logistic regression", LRC, "^"),
                                ("LR + Bayes-risk rule", "#2c4a70", "v"), ("Proposed CS-CART", ACC, "o")]):
    g = cq_t[cq_t.method == m]
    ax.errorbar(g.quartile, g["mean"], yerr=g["std"], marker=mk, color=c, capsize=2, lw=1, ms=4,
                label=m.replace("Logistic regression", "LR").replace("Proposed ", ""))
ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
ax.set_xlabel("Control-plane congestion quartile"); ax.set_ylabel("Recovery cost (min/call)")
ax.legend(frameon=False, ncol=2, fontsize=6.5)
save(fig, "fig6_congestion")

# ----------------------------------------------------------------------------------------------
metrics = dict(
    n_calls=int(n), folds=5, seed=SEED, oracle_cost_min=float(oracle_cost),
    best_fixed_baseline=best_base,
    main={m: {k: float(main.loc[m, k]) for k in main.columns} for m in ORDER},
    ablation={m: {k: float(abl_t.loc[m, k]) for k in abl_t.columns} for m in abl_t.index},
    recall_proposed=recall, significance=sig.to_dict(orient="records"),
    selected_hparams=hp_rows, inner_cv_best_cost_by_gamma=gamma_curve,
    ece_by_quartile={m: cq_t[cq_t.method == m].set_index("quartile")["mean"].to_dict()
                     for m in ["ECE CS-CART", "ECE LR"]},
    sensitivity_penalty=sens.groupby("scale").mean().drop(columns="fold").to_dict(orient="index"),
    top_features=imp.head(10).to_dict(), cost_params=P,
    mean_describe_delay_s=float(df.loc[df.describe_shows_resource >= 0, "describe_poll_delay_s"].mean()),
    runtime_s=time.time() - T0)
with open(os.path.join(RES, "metrics.json"), "w") as fh:
    json.dump(metrics, fh, indent=1, default=float)
print(main[["cost_min_mean", "cost_min_std", "inconsistent_per1k_mean", "disrupt_per1k_mean", "wait_rate_mean", "map_outcome_macro_f1_mean", "action_macro_f1_mean",
            "outcome_macro_f1_mean", "abort_rate_mean"]].round(3).to_string())
print(abl_t[["cost_min_mean", "inconsistent_per1k_mean", "action_macro_f1_mean", "describe_latency_s_mean"]]
      .round(3).to_string())
print(sig.round(4).to_string())
print(f"runtime {time.time() - T0:.0f}s")
