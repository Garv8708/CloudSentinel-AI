from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# Common label names found in classification datasets. If needed, set this
# to the exact label column name from your dataset.
TARGET_COLUMN = None
CANDIDATE_TARGET_COLUMNS = [
    "Attack_type", "attack_type", "label", "Label", "class", "Class", "target", "Target"
]
MAX_UPLOAD_MB = 20
