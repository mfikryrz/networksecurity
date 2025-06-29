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

---

Saya akan jelaskan code `data_transformation.py` ini secara rinci:

## Fungsi Utama Data Transformation

Code ini bertanggung jawab untuk **mentransformasi data** yang sudah divalidasi menjadi format yang siap untuk training model. Berikut penjelasan detailnya:

### 1. **Inisialisasi Class**
```python
def __init__(self, data_validation_artifact, data_transformation_config):
```
- Menerima hasil dari tahap validasi data sebelumnya
- Menerima konfigurasi untuk transformasi data

### 2. **Membaca Data**
```python
@staticmethod
def read_data(file_path) -> pd.DataFrame:
```
- Method static untuk membaca file CSV
- Mengembalikan pandas DataFrame

### 3. **Membuat Transformer Object**
```python
def get_data_transformer_object(cls) -> Pipeline:
```
- Membuat **KNNImputer** untuk mengisi missing values
- KNNImputer menggunakan algoritma K-Nearest Neighbors untuk mengisi nilai yang hilang berdasarkan nilai tetangga terdekat
- Dibungkus dalam Pipeline scikit-learn untuk memudahkan penggunaan

### 4. **Proses Transformasi Data**
```python
def initiate_data_transformation() -> DataTransformationArtifact:
```

Ini adalah method utama yang melakukan beberapa tahap:

#### a) **Membaca Data Training & Testing**
```python
train_df = DataTransformation.read_data(valid_train_file_path)
test_df = DataTransformation.read_data(valid_test_file_path)
```

#### b) **Memisahkan Features dan Target**
```python
# Training
input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
target_feature_train_df = train_df[TARGET_COLUMN]
target_feature_train_df = target_feature_train_df.replace(-1, 0)
```
- Memisahkan kolom input (features) dan output (target)
- **Mengganti nilai -1 menjadi 0** pada target (kemungkinan untuk binary classification)

#### c) **Fit dan Transform Data**
```python
preprocessor_object = preprocessor.fit(input_feature_train_df)
transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
```
- **Fit**: Preprocessor belajar dari data training (misal: mencari tetangga untuk KNN)
- **Transform**: Menerapkan transformasi ke data training dan testing

#### d) **Menggabungkan Features dan Target**
```python
train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
```
- Menggunakan `np.c_` untuk menggabungkan kolom features dan target
- Hasil akhir: numpy array yang siap untuk model training

#### e) **Menyimpan Hasil**
```python
save_numpy_array_data(transformed_train_file_path, array=train_arr)
save_numpy_array_data(transformed_test_file_path, array=test_arr)
save_object(transformed_object_file_path, preprocessor_object)
save_object("final_model/preprocessor.pkl", preprocessor_object)
```
- Menyimpan array training dan testing yang sudah ditransformasi
- Menyimpan preprocessor object untuk digunakan saat inference
- Menyimpan copy ke folder `final_model` untuk deployment

## Alur Transformasi

```
Validated Data → Read CSV → Separate Features/Target → 
→ Replace -1 with 0 → Fit KNNImputer → Transform Data → 
→ Combine Arrays → Save Arrays & Preprocessor
```

## Kegunaan dalam ML Pipeline

1. **Handling Missing Values**: Menggunakan KNNImputer untuk mengisi nilai yang kosong
2. **Data Preprocessing**: Mempersiapkan data dalam format numpy array
3. **Binary Classification Setup**: Mengubah label -1 menjadi 0 (kemungkinan dari {-1, 1} menjadi {0, 1})
4. **Reusability**: Menyimpan preprocessor untuk digunakan pada data baru saat production

## Output (DataTransformationArtifact)

- **transformed_object_file_path**: Path ke file preprocessor
- **transformed_train_file_path**: Path ke file training array
- **transformed_test_file_path**: Path ke file testing array

Code ini merupakan tahap penting sebelum model training, memastikan data dalam format yang konsisten dan tidak memiliki missing values.