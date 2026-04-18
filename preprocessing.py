import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer


FEATURE_COLS = ["Age", "Annual Income (k$)", "Spending Score", "Purchase Freq/mo", "Tenure (yrs)"]


def preprocess(df: pd.DataFrame, scaler_type: str = "standard") -> tuple[np.ndarray, object]:
    """
    Clean and scale features.

    Args:
        df: Raw customer DataFrame.
        scaler_type: 'standard' (Z-score) or 'minmax'.

    Returns:
        (scaled_array, fitted_scaler)
    """
    features = df[FEATURE_COLS].copy()

    # Impute missing values with column median
    imputer = SimpleImputer(strategy="median")
    features_imputed = imputer.fit_transform(features)

    # Scale
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled = scaler.fit_transform(features_imputed)
    return scaled, scaler


def get_feature_names() -> list[str]:
    return FEATURE_COLS
