import os, sys, time, json, gc, shutil
import multiprocessing as mp
import pandas as pd, psutil, lightgbm as lgb

sys.path.insert(0, "/kaggle/working/repo")
from src.blocking import build_index, prepare_pool
from src.features import add_relative_features
from src.matcher import train_matcher, macro_f05
from src.pipeline import build_pairs, predict_chunk

# WHY a smoke mode: a bug found after 1 hour of work is expensive, found in 3 minutes it is cheap
SMOKE = os.environ.get("SMOKE") == "1"
R = "/kaggle/input/datasets/gowthamdd/amazon-ml-challenge-2026/train/"
M = "/kaggle/working/repo/models/dev/"
W = "/kaggle/working/smoke/" if SMOKE else "/kaggle/working/full/"
CHUNK = 2_000 if SMOKE else 20_000
N_TRAIN = 10_000 if SMOKE else 300_000
WORKERS = 2          # WHY 2: measured about 5 GB RAM per worker, 4 would not fit
COUNTRIES = ["US", "India"]
ST, WB = {}, {}      # WHY globals: forked workers read them without pickling

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)   # WHY flush: the background log must show progress live

def chunks(n):
    return [(i, s, min(s + CHUNK, n)) for i, s in enumerate(range(0, n, CHUNK))]

def make_ids(gt):
    # WHY same calls as the laptop: the hold-out must be the same 300k entities
    dev = set(gt[["source1_entity_id", "matched_entity_ids"]].sample(100_000, random_state=42)["source1_entity_id"])
    hold = set(gt[~gt["source1_entity_id"].isin(dev)].sample(300_000, random_state=7)["source1_entity_id"])
    # WHY exclude hold: the final matcher must never train on the entities we score
    train = set(gt[~gt["source1_entity_id"].isin(hold)].sample(N_TRAIN, random_state=11)["source1_entity_id"])
    return hold, train

def s1_run(s1, country):
    d = s1[s1["country"] == country]
    return d.head(2 * CHUNK) if SMOKE else d

def build_country(country):
    # WHY free first: two indexes in RAM at once would waste about 12 GB
    ST.pop("index", None); ST.pop("prepared", None); gc.collect()
    parts = []
    for f in ["train_source2.tsv", "train_source3.tsv"]:
        for ch in pd.read_csv(R + f, sep="\t", chunksize=500_000):
            parts.append(ch[ch["country"] == country])
            if SMOKE:
                break
    pool = pd.concat(parts, ignore_index=True)
    t0 = time.time()
    gc.disable()      # WHY: millions of small lists trigger many useless collections
    ST["index"], ST["prepared"] = build_index(pool), prepare_pool(pool)
    gc.enable()
    del pool, parts; gc.collect()
    log(country, "index built:", len(ST["index"]), "keys |", round(time.time() - t0), "s")

def init_worker():
    # WHY own booster per worker: a forked child must not reuse the parent's OpenMP threads
    if ST.get("model_path"):
        WB["b"] = lgb.Booster(model_file=ST["model_path"])

def train_job(job):
    i, a, b = job
    path = f"{W}pairs/{ST['country']}_{i:04d}.pkl"
    if os.path.exists(path):          # WHY skip: a restarted run keeps finished chunks
        return i
    pr = build_pairs(ST["s1"].iloc[a:b], ST["index"], ST["prepared"])
    pr["label"] = [int(c in ST["truth"][s]) for s, c in zip(pr["s1"], pr["cand"])]
    pr = add_relative_features(pr)
    num = pr.columns.difference(["s1", "cand", "label"])
    pr[num] = pr[num].astype("float32")
    pr.to_pickle(path)
    return i

def predict_job(job):
    i, a, b = job
    path = f"{W}pred/{ST['country']}_{i:04d}.pkl"
    if os.path.exists(path):
        return i
    predict_chunk(ST["s1"].iloc[a:b], ST["index"], ST["prepared"], WB["b"], ST["cfg"]).to_pickle(path)
    return i

def run_pool(fn, jobs, tag):
    t0 = time.time()
    # WHY maxtasksperchild: a fresh worker shares the index pages again, so copied memory is released
    with mp.get_context("fork").Pool(WORKERS, initializer=init_worker, maxtasksperchild=8) as pool:
        for n, _ in enumerate(pool.imap_unordered(fn, jobs), 1):
            if n % 5 == 0 or n == len(jobs):
                log(f"{tag}: {n}/{len(jobs)} chunks | free RAM GB {psutil.virtual_memory().available / 1e9:.1f}")
    log(tag, "done in", round(time.time() - t0), "s")

def main():
    for d in ("pairs", "pred"):
        os.makedirs(W + d, exist_ok=True)
    cfg = json.load(open(M + "matcher_dev_v4_config.json"))
    cfg["threshold"] = 0.5   # WHY low: keep more rows, so the final threshold is chosen afterwards
    gt = pd.read_csv(R + "train_ground_truth.tsv", sep="\t")
    s1 = pd.read_csv(R + "train_source1.tsv", sep="\t")
    hold_ids, train_ids = make_ids(gt)
    log("smoke" if SMOKE else "FULL", "| hold", len(hold_ids), "| train", len(train_ids))

    # ---- A: training pairs, one country at a time ----
    if not os.path.exists(W + "model.txt"):
        gt_t = gt[gt["source1_entity_id"].isin(train_ids)]
        ST["truth"] = {s: (set(m.split(",")) if isinstance(m, str) else set())
                       for s, m in zip(gt_t["source1_entity_id"], gt_t["matched_entity_ids"])}
        for c in COUNTRIES:
            if os.path.exists(f"{W}pairs/{c}.done"):
                continue
            build_country(c)
            ST["country"] = c
            ST["s1"] = s1[(s1["country"] == c) & s1["entity_id"].isin(train_ids)].reset_index(drop=True)
            run_pool(train_job, chunks(len(ST["s1"])), f"pairs {c}")
            open(f"{W}pairs/{c}.done", "w").close()

        # ---- B: one matcher for both countries ----
        pairs = pd.concat([pd.read_pickle(f"{W}pairs/{f}") for f in sorted(os.listdir(W + "pairs"))
                           if f.endswith(".pkl")], ignore_index=True)
        log("training rows:", len(pairs), "| positive rate:", round(pairs["label"].mean(), 3))
        model = train_matcher(pairs, cfg["features"])
        model.booster_.save_model(W + "model.txt")
        del pairs, model; gc.collect()
        shutil.rmtree(W + "pairs")   # WHY: the pair files are large and not needed after training
        log("model saved")

    # ---- C: predict all S1 entities, one country at a time ----
    ST["model_path"], ST["cfg"] = W + "model.txt", cfg
    for c in COUNTRIES:
        if os.path.exists(f"{W}pred/{c}.done"):
            continue
        build_country(c)
        ST["country"], ST["s1"] = c, s1_run(s1, c).reset_index(drop=True)
        run_pool(predict_job, chunks(len(ST["s1"])), f"predict {c}")
        open(f"{W}pred/{c}.done", "w").close()

    # ---- D: score the hold-out entities ----
    kept = pd.concat([pd.read_pickle(f"{W}pred/{f}") for f in sorted(os.listdir(W + "pred"))
                      if f.endswith(".pkl")], ignore_index=True)
    ran = pd.concat([s1_run(s1, c) for c in COUNTRIES])
    country = dict(zip(ran["entity_id"], ran["country"]))
    ids = sorted(hold_ids & set(country))
    # WHY split: the threshold is chosen on the first third, the number we report comes from the rest
    tune, rep = ids[:len(ids) // 3], ids[len(ids) // 3:]
    gt_h = gt[gt["source1_entity_id"].isin(ids)]
    truth = {s: (set(m.split(",")) if isinstance(m, str) else set())
             for s, m in zip(gt_h["source1_entity_id"], gt_h["matched_entity_ids"])}

    def final_pred(t):
        k = kept[kept["p"] >= t].sort_values("p", ascending=False).drop_duplicates("cand")
        return k.groupby("s1")["cand"].apply(set).to_dict()

    scores = {t: macro_f05(final_pred(t), truth, tune) for t in [0.5, 0.6, 0.7, 0.8]}
    log("tune scores:", {t: round(v, 4) for t, v in scores.items()})
    best = max(scores, key=scores.get) # selecting key
    pred = final_pred(best)

    tp = n_pred = n_true = single = single_false = 0
    for e in rep:
        p, t = pred.get(e, set()), truth[e]
        if not t:
            single += 1
            single_false += bool(p)
            continue
        tp += len(p & t); n_pred += len(p); n_true += len(t)
    res = {"threshold": best, "n_report": len(rep),
           "macro_f05": round(macro_f05(pred, truth, rep), 4),
           "precision": round(tp / max(n_pred, 1), 4), "recall": round(tp / max(n_true, 1), 4),
           "false_singletons": f"{single_false} of {single}"}
    for c in COUNTRIES:
        sub = [e for e in rep if country[e] == c]
        if sub:
            res[c] = round(macro_f05(pred, truth, sub), 4)
    json.dump(res, open(W + "results.json", "w"), indent=2)
    log("RESULTS", res)

if __name__ == "__main__": #it is main script
    main()