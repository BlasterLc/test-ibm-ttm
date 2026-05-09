import pandas as pd
import torch

from tsfm_public import TimeSeriesPreprocessor
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from transformers import Trainer, TrainingArguments
from tsfm_public.toolkit.dataset import ForecastDFDataset

def run_finetune():
    df = pd.read_csv("data/raw/sales.csv", parse_dates=["date"])

    tsp = TimeSeriesPreprocessor(
    timestamp_column="date",
    target_columns=["prod_1","prod_2","prod_3","prod_4",
                    "prod_5","prod_6","prod_7","prod_8","prod_9"],
    scaling=True,
    )   

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    tsp.train(df.iloc[:train_end])


    train_dataset = ForecastDFDataset(
        tsp.preprocess(df.iloc[:train_end]),
        timestamp_column="date",
        target_columns=["prod_1","prod_2","prod_3","prod_4",
                        "prod_5","prod_6","prod_7","prod_8","prod_9"],
        context_length=512,
        prediction_length=96
    )

    val_dataset = ForecastDFDataset(
        tsp.preprocess(df.iloc[train_end:val_end]),
        timestamp_column="date",
        target_columns=["prod_1","prod_2","prod_3","prod_4",
                        "prod_5","prod_6","prod_7","prod_8","prod_9"],
        context_length=512,
        prediction_length=96
    )


    test_dataset = ForecastDFDataset(
        tsp.preprocess(df.iloc[val_end:]),
        timestamp_column="date",
        target_columns=["prod_1","prod_2","prod_3","prod_4",
                        "prod_5","prod_6","prod_7","prod_8","prod_9"],
        context_length=512,
        prediction_length=96
    )


    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        prediction_length=96
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Usando: {device} para entrenamiento")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./models",
            num_train_epochs=50,
            per_device_train_batch_size=8,
            learning_rate=0.001,
            save_strategy="no",
            logging_steps=10,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    #guardar el modelo

    model.save_pretrained("./models/finetuned_ttm")
    tsp.save_pretrained("./models/ttm_helix")
    print("Modelo guardado en ./models/ttm_helix")

if __name__ == "__main__":
    run_finetune()
        