import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
if __name__=='__main__':
    FILE_PATH="Network_Data\phisingData.csv"
    DATABASE="mfikryrz"
    Collection="NetworkData"
    networkobj=NetworkDataExtract()
    records=networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)
        


"""
Tentu! Berikut penjelasan lengkap dan terstruktur dari file `push_data.py` yang kamu buat:

---

## Tujuan Umum Script `push_data.py`

Script ini bertugas untuk:

1. **Membaca data CSV** berisi data keamanan jaringan.
2. **Mengonversi data tersebut ke format JSON.**
3. **Menghubungkan ke MongoDB Atlas menggunakan URI dari `.env`.**
4. **Menyimpan data ke dalam database MongoDB.**

---

## Penjelasan Per Bagian

### 1. **Load Environment Variable**

```python
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)
```

* Mengambil `MONGO_DB_URL` dari file `.env`, yang menyimpan URI koneksi ke MongoDB Atlas.
* Berguna agar kamu tidak menuliskan kredensial sensitif secara langsung di dalam kode.

---

### 2. **Impor dan Setup Tambahan**

```python
import certifi
ca = certifi.where()
```

* `certifi` digunakan untuk mencari lokasi sertifikat SSL terpercaya agar koneksi HTTPS ke MongoDB bisa aman.
* Namun, kamu belum memanfaatkannya di `MongoClient(...)`. Kalau ingin aman SSL, bisa tambahkan `tlsCAFile=ca`.

---

### 3. **Impor Modul Lain**

```python
import pandas as pd
import json
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
```

* `pandas` dan `json`: untuk memproses CSV → JSON
* `pymongo`: untuk berinteraksi dengan MongoDB
* `NetworkSecurityException`: untuk logging error dengan konteks lokasi kesalahan
* `logger`: untuk mencatat log (logging) ke file `.log`

---

### 4. **Class `NetworkDataExtract`**

Ini adalah class utama yang berisi dua metode:

#### a. `csv_to_json_convertor`

```python
def csv_to_json_convertor(self, file_path):
    data = pd.read_csv(file_path)
    data.reset_index(drop=True, inplace=True)
    records = list(json.loads(data.T.to_json()).values())
    return records
```

* Membaca file CSV
* Reset index agar bersih
* Konversi ke `list of dictionaries` menggunakan `json.loads(...data.to_json...)` untuk kemudian disimpan ke MongoDB

#### b. `insert_data_mongodb`

```python
def insert_data_mongodb(self, records, database, collection):
    self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
    self.database = self.mongo_client[database]
    self.collection = self.database[collection]
    self.collection.insert_many(records)
    return len(records)
```

* Membuat koneksi ke MongoDB menggunakan URI
* Memilih database dan collection
* Menyisipkan seluruh `records` (list of dict)
* Mengembalikan jumlah record yang berhasil dimasukkan

---

### 5. **Main Script**

```python
if __name__=='__main__':
    FILE_PATH = "Network_Data\phisingData.csv"
    DATABASE = "mfikryrz"
    Collection = "NetworkData"

    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)
```

* Menentukan path ke file data dan target MongoDB.
* Membuat objek dari class.
* Menjalankan proses konversi dan upload ke MongoDB.

---

## Kesimpulan Singkat

| Komponen                   | Fungsi                                               |
| -------------------------- | ---------------------------------------------------- |
| `.env`                     | Menyimpan koneksi MongoDB (rahasia)                  |
| `csv_to_json_convertor`    | Ubah CSV ke JSON (list of dict) untuk MongoDB        |
| `insert_data_mongodb`      | Simpan data ke collection MongoDB yang ditentukan    |
| `NetworkSecurityException` | Custom error handler untuk memudahkan debug          |
| `certifi`                  | (Opsional) Digunakan untuk koneksi TLS/SSL yang aman |

---

## Saran Tambahan

1. **Tambahkan TLS/SSL saat koneksi MongoDB:**

   ```python
   self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
   ```

2. **Validasi file CSV:** agar tidak crash kalau file kosong atau rusak.

3. **Gunakan logger saat mencetak info penting**, contoh:

   ```python
   logging.info(f"{no_of_records} records inserted into {DATABASE}.{Collection}")
   ```

---

Kalau kamu mau, aku bisa bantu refactor file ini agar lebih modular atau siap untuk production.

"""