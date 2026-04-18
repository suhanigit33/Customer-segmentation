import numpy as np
import pandas as pd

def generate_customer_data(n_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data for segmentation.

    Features:
        - CustomerID
        - Age
        - Annual Income (k$)
        - Spending Score (1-100)
        - Purchase Frequency (per month)
        - Tenure (years as customer)
    """
    rng = np.random.default_rng(random_state)

    # Cluster centers: [age, income, spend, freq, tenure]
    centers = [
        [32, 88, 78, 20, 3],   # Young high-earners, big spenders
        [55, 45, 30, 6,  8],   # Middle-aged, conservative
        [28, 35, 65, 15, 2],   # Young, low income but high spend
        [48, 72, 45, 10, 6],   # Mature, moderate spend
    ]
    scales = [8, 12, 15, 5, 2]
    ranges = [(18, 70), (15, 130), (1, 100), (1, 30), (0, 15)]

    rows = []
    per_cluster = n_samples // len(centers)
    for c in centers:
        for _ in range(per_cluster):
            row = [
                np.clip(rng.normal(c[i], scales[i]), ranges[i][0], ranges[i][1])
                for i in range(len(c))
            ]
            rows.append(row)

    # Remainder
    for _ in range(n_samples - len(rows)):
        c = centers[rng.integers(len(centers))]
        row = [np.clip(rng.normal(c[i], scales[i]), ranges[i][0], ranges[i][1]) for i in range(len(c))]
        rows.append(row)

    df = pd.DataFrame(rows, columns=["Age", "Annual Income (k$)", "Spending Score", "Purchase Freq/mo", "Tenure (yrs)"])
    df.insert(0, "CustomerID", range(1, len(df) + 1))
    df = df.round(1)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)
