import pandas as pd

def load_sales(df: pd.DataFrame) -> pd.DataFrame:

    df.index = pd.to_datetime(df.index)

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")

    df = df.reindex(full_range)

    df = df.fillna(0)

    df.index.name = "date"

    return df