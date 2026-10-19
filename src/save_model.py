import s3fs
import joblib

from typing import Any
def save_model(
        bucket_key: str,
        model: Any,
        fs: s3fs.S3FileSystem,
) -> None:
    with fs.open(path=bucket_key, mode="wb") as stream:
        joblib.dump(model, stream)
        print("Model saved successfully!")
