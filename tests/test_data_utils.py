import pandas as pd
from src.data_utils import split_features_target

def test_split_features_target():
    df = pd.DataFrame({
        "duration": [1, 2, 3, 4],
        "protocol": ["tcp", "udp", "tcp", "udp"],
        "label": ["normal", "attack", "normal", "attack"]
    })
    X, y, target = split_features_target(df)
    assert target == "label"
    assert list(X.columns) == ["duration", "protocol"]
    assert y.tolist() == ["normal", "attack", "normal", "attack"]
