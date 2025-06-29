'''
The setup.py file is an essential part of packaging and 
distributing Python projects. It is used by setuptools 
(or distutils in older Python versions) to define the configuration 
of your project, such as its metadata, dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    Thiss function will return list of requirements
    
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="mfikryrizal",
    author_email="muh.fikryrizal@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

"""
Tentu! Berikut adalah penjelasan lengkap dari isi file `setup.py` milikmu:

---

## Tujuan Umum File `setup.py`

File `setup.py` adalah file konfigurasi yang digunakan untuk:

* **Mendistribusikan** dan **mengemas** proyek Python kamu menjadi sebuah **package**.
* Dapat di-*install* seperti `pip install .`
* Mendefinisikan informasi metadata (nama, versi, author, dependency, dsb).

---

## Penjelasan Per Baris

### 1. **Import Library**

```python
from setuptools import find_packages, setup
from typing import List
```

* `setuptools`: library standar Python untuk packaging.
* `find_packages()`: otomatis mencari semua subfolder yang berisi `__init__.py` (artinya package Python).
* `List` dari `typing`: digunakan untuk tipe anotasi fungsi.

---

### 2. **Fungsi `get_requirements()`**

```python
def get_requirements()->List[str]:
```

Fungsi ini bertugas membaca `requirements.txt` dan mengembalikan daftar dependensi.

```python
    requirement_lst:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
```

* Menghapus whitespace dan mengabaikan baris kosong atau `-e .`
* `-e .` biasanya digunakan untuk *editable install*, tapi tidak dimasukkan dalam `install_requires`.

---

### 3. **Fungsi `setup()`**

```python
setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="mfikryrizal",
    author_email="muh.fikryrizal@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
```

Penjelasan setiap argumen:

* `name`: nama package kamu (`NetworkSecurity`)
* `version`: versi release pertama (`0.0.1`)
* `author` dan `author_email`: informasi kamu sebagai pembuat package
* `packages=find_packages()`: otomatis menyertakan semua subfolder yang merupakan package
* `install_requires=get_requirements()`: menyisipkan semua dependensi dari `requirements.txt`

---

## Bagaimana File Ini Digunakan?

Jika kamu menjalankan:

```bash
pip install .
```

Python akan:

* Membaca `setup.py`
* Menjalankan `get_requirements()` untuk meng-install dependensi
* Mengemas semua folder Python (seperti `networksecurity/`) ke dalam package
* Menjadikan proyekmu bisa di-*import* sebagai package Python

Contoh:

```python
from networksecurity.logging import logger
```

---

## Ringkasan

| Elemen                | Fungsi                                                               |
| --------------------- | -------------------------------------------------------------------- |
| `get_requirements()`  | Membaca file `requirements.txt` dan mengabaikan `-e .`               |
| `setup(...)`          | Mendaftarkan metadata dan konfigurasi untuk membentuk package Python |
| `find_packages()`     | Otomatis menyertakan semua submodul di dalam `networksecurity/`      |
| File ini dipakai oleh | `pip`, `build`, `twine`, dan sistem packaging PyPI                   |

---

Kalau kamu ingin mempublikasikan project ini ke **PyPI** atau membagikannya ke rekan developer lain, `setup.py` ini adalah fondasi penting.

Jika kamu ingin bantuan untuk membuat file `pyproject.toml` atau modernisasi package agar sesuai standar baru Python, aku juga bisa bantu.


"""