import pandas as pd
import torch
import numpy as np
from tsfm_public import TimeSeriesPreprocessor
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.dataset import ForecastDFDataset
from torch.utils.data import DataLoader

def run_predict():
    df = pd.read_csv("data/raw/sales.csv", parse_dates=["date"])

    tsp = TimeSeriesPreprocessor.from_pretrained("./models/ttm_helix")

    context = df.tail(512)

    dataset = ForecastDFDataset(
        tsp.preprocess(context),
        timestamp_column="date",
        target_columns=["prod_1","prod_2","prod_3","prod_4",
                        "prod_5","prod_6","prod_7","prod_8","prod_9"],
        context_length=512,
        prediction_length=96
    )

    model = TinyTimeMixerForPrediction.from_pretrained("./models/finetuned_ttm")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    batch = next(iter(loader))

    with torch.no_grad():
        output = model(past_values=batch["past_values"].to(device))

    predictions = output.prediction_outputs[0].cpu().numpy()

    target_cols = ["prod_1","prod_2","prod_3","prod_4",
                   "prod_5","prod_6","prod_7","prod_8","prod_9"]

    pred_df = pd.DataFrame(predictions, columns=target_cols)
    pred_df = tsp.inverse_scale_targets(pred_df)
    predictions = pred_df.values.clip(min=0)
    
    
    #mostrar resultados

    last_dates = df["date"].max()
    future_dates = pd.date_range(start=last_dates + pd.Timedelta(days=1), periods=96)

    result = pd.DataFrame(predictions, columns=["prod_1","prod_2","prod_3","prod_4",
                                                "prod_5","prod_6","prod_7","prod_8","prod_9"])
    
    result.insert(0, "date", future_dates)
    result = result.round(2)

    print("\n=== Predicciones para los próximos 96 días ===")
    print(result.to_string(index=False))


    print("\n=== Predicción completada ===")
    print("\n=== Promedio diario predicho por producto ===")
    print(result.drop(columns="date").mean().round(2))

    
def collate_fn(batch):
    return {
        k: torch.stack([torch.tensor(np.array(item[k])) for item in batch])
        for k in batch[0] if not isinstance(batch[0][k], pd.Timestamp)
    }



if __name__ == "__main__":
    run_predict()
