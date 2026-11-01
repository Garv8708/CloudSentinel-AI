
import pandas as pd

PATH = "data/network_traffic.csv"
TARGET = "Attack_type"

print("Loading dataset...")
df = pd.read_csv(PATH, low_memory=False)

print("\n--- DATASET SUMMARY ---")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", int(df.isna().sum().sum()))
print("Exact duplicate rows:", int(df.duplicated().sum()))

if TARGET not in df.columns:
    raise ValueError(
        f"Target column {TARGET!r} not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("\n--- CLASS DISTRIBUTION ---")
print(df[TARGET].value_counts(dropna=False).to_string())

print("\n--- POSSIBLE ID / INDEX COLUMNS ---")
for col in df.columns:
    name = str(col).strip().lower()
    if (
        name in {"id", "index", "unnamed: 0", "row_id", "record_id"}
        or name.startswith("unnamed:")
    ):
        print(
            col,
            "| unique values:", df[col].nunique(dropna=True),
            "| sample:", df[col].head(5).tolist()
        )

print("\n--- COLUMNS THAT MAY ENCODE THE TARGET ---")
for col in df.columns:
    if col == TARGET:
        continue
    name = str(col).strip().lower()
    if any(word in name for word in ["attack", "label", "target", "class"]):
        print(
            col,
            "| unique values:", df[col].nunique(dropna=True),
            "| sample:", df[col].dropna().astype(str).unique()[:10].tolist()
        )

print("\n--- EXACT FEATURE DUPLICATES ACROSS DIFFERENT LABELS ---")
features = df.drop(columns=[TARGET])
valid = df[TARGET].notna()
tmp = features.loc[valid].copy()
tmp["_target_for_audit"] = df.loc[valid, TARGET].astype(str).values

conflicting = (
    tmp.groupby(list(features.columns), dropna=False)["_target_for_audit"]
    .nunique()
    .gt(1)
)

print(
    "Distinct feature combinations associated with multiple labels:",
    int(conflicting.sum())
)

print("\nAudit complete. No data has been modified.")
