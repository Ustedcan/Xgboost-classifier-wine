import kagglehub
import os
import pandas as pd
import s3fs

def get_kaggle_dataset(
        kaggle_path: str,
        raw_file_name: str,
        bucket_key: str,
        fs: s3fs.S3FileSystem
) -> None:
    path = kagglehub.dataset_download(kaggle_path)
    lpath = os.path.join(path, raw_file_name)
    fs.put_file(lpath=lpath, rpath=bucket_key)

def load_file_s3(
        bucket_key: str,
        fs: s3fs.S3FileSystem
) -> pd.DataFrame:
    with fs.open(bucket_key, mode="rb") as stream:
        return pd.read_csv(stream)

def save_file_s3(
        df: pd.DataFrame,
        bucket_key: str,
        fs: s3fs.S3FileSystem
) -> None:
    with fs.open(bucket_key, mode="wb") as stream:
        df.to_csv(stream)
    


