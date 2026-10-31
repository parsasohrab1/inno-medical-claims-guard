"""Anomaly scoring of insurance claims with human-readable reasons.

Lab prototype on SYNTHETIC data only (TRL 4). The score only prioritises human review; it never decides a claim.
"""
import numpy as np

N_CLAIMS, N_CODES, N_CENTERS = 4000, 40, 25

CRITERIA = [
    ("auc", ">=", 0.90, "ROC AUC of anomaly score"),
    ("precision_at_5pct", ">=", 0.60, "precision among top 5% scored claims"),
    ("explained_share", ">=", 1.0, "flagged claims that carry at least one reason"),
]


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    price = np.exp(rng.normal(np.log(400_000), 0.9, N_CODES))
    claims = []
    for i in range(N_CLAIMS):
        code = int(rng.integers(0, N_CODES)); qty = int(rng.choice([1, 1, 1, 2, 3]))
        claims.append(dict(claim_id="CL%05d" % i, center="H%02d" % rng.integers(0, N_CENTERS), code="S%02d" % code,
                           patient="P%04d" % rng.integers(0, 1500), day=int(rng.integers(0, 365)), qty=qty,
                           amount=float(price[code] * qty * np.exp(rng.normal(0, 0.08))), anomaly="", is_anomaly=0))
    idx = rng.choice(N_CLAIMS, int(0.05 * N_CLAIMS), replace=False)
    for j, i in enumerate(idx):
        c = claims[i]; kind = ["upcoding", "duplicate", "excess_qty"][j % 3]
        if kind == "upcoding":
            c["amount"] *= rng.uniform(2.0, 4.0)
        elif kind == "excess_qty":
            c["amount"] *= (n := int(rng.integers(10, 20))) / c["qty"]; c["qty"] = n
        else:
            src = claims[int(rng.integers(0, N_CLAIMS))]
            for k in ("center", "code", "patient", "day", "qty", "amount"):
                c[k] = src[k]
            if src is c:
                kind = "upcoding"; c["amount"] *= 3
        c["anomaly"] = kind; c["is_anomaly"] = 1
    return claims


def score(claims):
    by_code = {}
    for c in claims:
        by_code.setdefault(c["code"], []).append(np.log(c["amount"] / c["qty"]))
    stats = {k: (np.median(v), max(1.4826 * np.median(np.abs(np.array(v) - np.median(v))), 0.05)) for k, v in by_code.items()}
    seen = {}; out = []
    for c in claims:
        reasons = []; s = 0.0
        med, mad = stats[c["code"]]
        z = (np.log(c["amount"] / c["qty"]) - med) / mad
        if z > 3:
            reasons.append("unit price %.1f robust-SD above the median for %s" % (z, c["code"])); s += min(z / 8, 1.0)
        if c["qty"] > 6:
            reasons.append("quantity %d is far above usual (1-3)" % c["qty"]); s += min(c["qty"] / 12, 1.0)
        key = (c["patient"], c["code"], c["day"])
        if key in seen:
            reasons.append("same patient, service and date already billed in %s" % seen[key]); s += 1.0
        else:
            seen[key] = c["claim_id"]
        out.append({"claim_id": c["claim_id"], "score": float(1 - np.exp(-1.5 * s)), "reasons": reasons})
    return out


def center_trend(claims, scored, thr=0.5):
    tot, bad = {}, {}
    for c, s in zip(claims, scored):
        tot[c["center"]] = tot.get(c["center"], 0) + 1
        bad[c["center"]] = bad.get(c["center"], 0) + (s["score"] >= thr)
    return sorted(((k, bad[k] / tot[k]) for k in tot), key=lambda x: -x[1]) if tot else []


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(seed=0):
    claims = simulate(seed); sc = score(claims)
    y = np.array([c["is_anomaly"] for c in claims]); s = np.array([x["score"] for x in sc])
    top = np.argsort(-s)[: int(0.05 * len(s))]
    flagged = [x for x in sc if x["score"] >= 0.5]
    return {"auc": auc(y, s), "precision_at_5pct": float(y[top].mean()),
            "explained_share": float(np.mean([bool(x["reasons"]) for x in flagged])) if flagged else 1.0}


def generate(seed=0):
    claims = simulate(seed); sc = score(claims)
    return {"claims": claims, "scores": [dict(claim_id=x["claim_id"], score=round(x["score"], 4), reasons="; ".join(x["reasons"])) for x in sc]}


def demo(seed=0):
    claims = simulate(seed); sc = score(claims)
    best = sorted(sc, key=lambda x: -x["score"])[:3]
    return "\n".join("%s score=%.2f %s" % (b["claim_id"], b["score"], b["reasons"]) for b in best)
