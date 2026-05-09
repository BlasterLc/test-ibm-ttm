import pandas as pd
import numpy as np

def generate_synthetic_data(output_path="data/raw/sales.csv"):
    np.random.seed(42)

    dates = pd.date_range(start="2024-01-01", periods=600, freq="D")

    def serie(high, zero_prob=0.3):
        vals = np.random.randint(1, high+1, size=600).astype(float)
        mask = np.random.random(600) < zero_prob
        vals[mask] = 0
        return vals


    df = pd.DataFrame({
        "date": dates,
        "prod_1": serie(5),
        "prod_2": serie(10, zero_prob=0.2),
        "prod_3": serie(6),
        "prod_4": serie(4),
        "prod_5": serie(2, zero_prob=0.5),
        "prod_6": serie(5),
        "prod_7": serie(6),
        "prod_8": serie(2, zero_prob=0.5),
        "prod_9": serie(8, zero_prob=0.2)
    })

    df.to_csv(output_path, index=False)
    print(f"CSV generado con {len(df)} filas en {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data()