from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


## configuration of the Data Ingestion Config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb
        """
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]

            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df=df.drop(columns=["_id"],axis=1)
            
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException
        
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException

"""
Saya akan jelaskan kegunaan dari code `data_ingestion.py` ini:

## Fungsi Utama Data Ingestion

Code ini adalah komponen untuk **mengambil dan mempersiapkan data** dalam sebuah pipeline machine learning untuk deteksi keamanan jaringan. Berikut penjelasan detailnya:

### 1. **Export Data dari MongoDB** 
```python
def export_collection_as_dataframe(self):
```
- Mengambil data dari database MongoDB 
- Mengkonversi collection MongoDB menjadi pandas DataFrame
- Menghapus kolom `_id` (ID internal MongoDB)
- Mengganti nilai "na" dengan `np.nan` untuk handling missing values

### 2. **Simpan Data ke Feature Store**
```python
def export_data_into_feature_store(self, dataframe: pd.DataFrame):
```
- Menyimpan DataFrame ke file CSV sebagai "feature store"
- Membuat folder jika belum ada
- Feature store ini berfungsi sebagai tempat penyimpanan data yang sudah diproses

### 3. **Split Data Training dan Testing**
```python
def split_data_as_train_test(self, dataframe: pd.DataFrame):
```
- Membagi data menjadi training set dan test set
- Menggunakan ratio yang sudah dikonfigurasi (misalnya 80:20)
- Menyimpan kedua set data ke file CSV terpisah

### 4. **Orchestrator Method**
```python
def initiate_data_ingestion(self):
```
Method utama yang menjalankan seluruh proses:
1. Ambil data dari MongoDB
2. Simpan ke feature store
3. Split menjadi train/test
4. Return artifact yang berisi path ke file training dan testing

## Alur Kerja

```
MongoDB → DataFrame → Feature Store (CSV) → Train/Test Split → Training & Testing Files
```

## Kegunaan dalam ML Pipeline

Komponen ini biasanya adalah **langkah pertama** dalam pipeline machine learning:
- **Input**: Database MongoDB dengan data jaringan
- **Output**: File training dan testing yang siap digunakan untuk tahap selanjutnya (data validation, transformation, model training)

Code ini menggunakan design pattern yang baik dengan:
- Configuration management (DataIngestionConfig)
- Artifact tracking (DataIngestionArtifact)
- Error handling dengan custom exception
- Logging untuk monitoring

Ini memudahkan untuk maintain dan scale sistem ML di production.
"""