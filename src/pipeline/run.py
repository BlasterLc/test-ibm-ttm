from src.pipeline.extract import extract_sales
from src.pipeline.transform import transform_sales
from src.pipeline.load import load_sales

def run_pipeline():
    print("Extrayendo datos de supabase")
    df = extract_sales()

    print("Transformando datos...")
    df = transform_sales(df)

    print("Cargando y rellenando fechas...")
    df = load_sales(df)

    print("Guardando CSV...")
    df.to_csv("data/raw/sales.csv")

    print(f"Pipeline completado. Shape: {df.shape}")

if __name__ == "__main__":
    run_pipeline()



