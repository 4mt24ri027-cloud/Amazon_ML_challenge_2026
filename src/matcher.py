import lightgbm as lgb

# WHY a fixed list: the model must see the same columns in the same order at train and predict time
FEATURES = ["name_missing", "name_set", "name_sort", "name_ratio", "name_tri", "name_tok",
            "addr_missing", "addr_jaccard", "addr_min", "num_match", "num_conflict",
            "word_jaccard"]


def train_matcher(train_pairs):
    # WHY LightGBM: handles 0/1 flags and 0-1 scores without scaling, fast on 2.3M rows
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                               random_state=42, verbose=-1)
    model.fit(train_pairs[FEATURES], train_pairs["label"])
    return model


def predict_matches(pairs, model, threshold=0.70):
    # WHY copy: we add a column and must not change the caller's table
    df = pairs[["s1", "cand"]].copy()
    df["p"] = model.predict_proba(pairs[FEATURES])[:, 1]
    # WHY threshold first: low-probability pairs should never claim a record
    df = df[df["p"] >= threshold]
    # WHY sort + drop_duplicates: ground truth never shares a record between S1 entities,
    # so each S2/S3 record goes only to the S1 entity that scores it highest
    df = df.sort_values("p", ascending=False).drop_duplicates("cand")
    return df.groupby("s1")["cand"].apply(set).to_dict()


def f_half(pred, true):
    # WHY same as the EDA notebook: a singleton scores 1.0 only if we predict nothing
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_by_s1, truth, s1_ids):
    # WHY loop over all s1_ids, not pred_by_s1: entities with no prediction must still be scored
    # (a singleton with no prediction is a perfect 1.0)
    scores = [f_half(pred_by_s1.get(e, set()), truth[e]) for e in s1_ids]
    return sum(scores) / len(scores)