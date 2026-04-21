import pandas as pd

def transform_sales(df: pd.DataFrame) -> pd.DataFrame:
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date

    unique_products = sorted(df["product_id"].unique())
    product_map = {pid: f"prod_{i+1}" for i, pid in enumerate(unique_products)}
    df["product_id"] = df["product_id"].map(product_map)

    df = df.groupby(["sale_date", "product_id"])["quantity"].sum().reset_index()
    
    df = df.pivot(index="sale_date", columns="product_id", values="quantity")
    df = df.fillna(0)
    df.index.name = "date"
    df.columns.name = None 

    return df
