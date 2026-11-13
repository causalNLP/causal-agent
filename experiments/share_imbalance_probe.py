"""List benchmark datasets by treated/untreated share (side-note helper)."""
import pandas as pd
import os

out = []
for split, f, d in [("qr", "experiments/data/causcibench/metadata_csv/qr_input.csv", "experiments/data/causcibench/qrdata"),
                    ("syn", "experiments/data/causcibench/metadata_csv/synthetic_input.csv", "experiments/data/causcibench/synthetic_data")]:
    m = pd.read_csv(f)
    for _, r in m.iterrows():
        t = str(r["treatment"]); ds = str(r["dataset_name"])
        p = os.path.join(d, ds)
        if not os.path.exists(p):
            p = os.path.join(d, f"cleaned_{ds}")
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            print("read fail", ds, exc)
            continue
        if t in df.columns:
            s = pd.to_numeric(df[t], errors="coerce")
            if s.notna().sum() == 0:
                continue
            out.append((split, int(r["id"]), ds[:28], t[:20], len(df), int((s == 1).sum()), int((s == 0).sum()), round(s.mean(), 3)))

res = pd.DataFrame(out, columns=["split", "id", "dataset", "treat_var", "n", "treated", "untreated", "treated_share"])
print("== most treated-heavy (few controls) ==")
print(res.nlargest(6, "treated_share").to_string(index=False))
print("\n== most control-heavy (few treated) ==")
print(res.nsmallest(6, "treated_share").to_string(index=False))
print("\n== datasets with zero untreated ==")
print(res[res["treated_share"] == 1][["split", "id", "dataset", "treat_var", "n", "treated"]].head(6).to_string(index=False))
print("total fully-treated datasets:", int((res["treated_share"] == 1).sum()))
