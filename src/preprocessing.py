import pandas as pd
import numpy as np

from typing import Tuple, Any
from sklearn.preprocessing import LabelEncoder

def drop_columns(
        df: pd.DataFrame
) -> pd.DataFrame:
    df = df.drop(columns=["Id"], errors="ignore")
    return df

def label_encoding(
        y_train: np.ndarray,
        y_test: np.ndarray,
        label_encoder: LabelEncoder
) -> Tuple[Any, Any]: # label enconding return an object MatrixLike or ArrayLike not numpy array
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    return y_train_encoded, y_test_encoded
    