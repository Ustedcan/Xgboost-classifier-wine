from typing import Tuple
import numpy as np
import pandas as pd

def get_x_y(
        df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    return x, y