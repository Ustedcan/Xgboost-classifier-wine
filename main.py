import os
import s3fs
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.get_kgg_dataset import get_kaggle_dataset, load_file_s3, save_file_s3
from src.preprocessing import drop_columns, label_encoding
from src.get_x_y import get_x_y
from src.save_model import save_model

BUCKET_FOLDER = "wine"
RAW_FILE_NAME = "WineQT.csv"
CLEAN_FILE_NAME = "wine_preprocessed.csv"

MODEL_FOLDER = "models"
MODEL_NAME = "xgboost_wine"

def main():

    load_dotenv()
    kaggle_dataset_path = os.getenv("KAGGLE_DATASET_PATH")
    bucket_bronze = os.getenv("BUCKET_BRONZE")
    bucket_silver = os.getenv("BUCKET_SILVER")
    bucket_gold = os.getenv("BUCKET_GOLD")

    bronze_key = f"{bucket_bronze}/{BUCKET_FOLDER}/{RAW_FILE_NAME}"
    silver_key = f"{bucket_silver}/{BUCKET_FOLDER}/{CLEAN_FILE_NAME}"
    gold_key = f"{bucket_gold}/{MODEL_FOLDER}"

    fs = s3fs.S3FileSystem()

    # Bajar el dataset de kaggle y guardarlo en S3
    get_kaggle_dataset(
        kaggle_path=kaggle_dataset_path, # type: ignore
        raw_file_name=RAW_FILE_NAME,
        bucket_key=bronze_key,
        fs=fs
    )

    # Cargar en memoria dataset
    df = load_file_s3(
        bucket_key=bronze_key,
        fs=fs
    )

    # Preprocessing
    df_cleaned = drop_columns(df)

    save_file_s3(
        df=df_cleaned,
        bucket_key=silver_key,
        fs=fs
    )

    x, y = get_x_y(
        df=df_cleaned
    )

    # Train test spli
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    label_encoder = LabelEncoder()

    y_train, y_test = label_encoding(
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder
    )

    # Model
    classifier = XGBClassifier(
        device = "cuda",
        n_estimators = 500,
        learning_rate = 0.05,
        random_state = 42,
        verbosity = 3
    )
    fitted_model = classifier.fit(x_train, y_train)
    y_pred = fitted_model.predict(x_test)
    #y_pred_og = label_encoder.inverse_transform(y_pred)

    report = classification_report(
        y_test, 
        y_pred,
        target_names=[str(cls) for cls in label_encoder.classes_]
    )
    print(report)

    # Save model
    model_path = f"{gold_key}/{MODEL_NAME}.joblib"
    save_model(
        bucket_key=model_path, # gold/models/xgboos_wine.joblib
        model=fitted_model,
        fs=fs
    )
        
if __name__ == "__main__":
    main()
