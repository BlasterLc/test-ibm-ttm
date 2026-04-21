import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def extract_sales() -> pd.DataFrame:
    client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY")
    )

    response = client.table("sales_transactions").select(
        "product_id, quantity, sale_date"
    ).execute()

    return pd.DataFrame(response.data)