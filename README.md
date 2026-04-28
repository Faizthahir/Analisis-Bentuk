# Shape Analysis Pipeline untuk Klasifikasi Objek

## 📌 Deskripsi

Proyek ini merupakan implementasi *shape analysis pipeline* untuk mengklasifikasikan objek berdasarkan karakteristik bentuknya. Pendekatan ini tidak menggunakan warna atau tekstur, melainkan fitur geometris seperti luas, perimeter, dan kontur objek.

Dataset terdiri dari tiga kelas objek, yaitu:

* Dompet
* Earphone
* Charger

Masing-masing kelas memiliki 6 citra dengan variasi posisi dan orientasi.

---

## ⚙️ Metode yang Digunakan

### 1. Preprocessing

* Konversi citra ke grayscale
* Segmentasi menggunakan *adaptive threshold*
* Ekstraksi kontur dengan `cv2.findContours()`
* Mengambil kontur terbesar sebagai objek utama

---

### 2. Ekstraksi Fitur

#### 🔹 Region Properties

* Area
* Perimeter
* Aspect Ratio
* Extent
* Solidity

#### 🔹 Moments

* Hu Moments (7 fitur invariant terhadap rotasi, translasi, dan skala)

#### 🔹 Fourier Descriptor

* Transformasi kontur ke domain frekuensi menggunakan FFT
* Digunakan untuk menangkap bentuk global objek

---

### 3. Klasifikasi

* Algoritma: **k-Nearest Neighbors (k-NN)**
* Parameter: k = 1 (cocok untuk dataset kecil)
* Evaluasi:

  * Train accuracy
  * Train-test split (stratified)
  * Cross-validation (3-fold)

---

## 📊 Hasil

| Metode Evaluasi     | Akurasi |
| ------------------- | ------- |
| Train (tanpa split) | 1.00    |
| Train-Test Split    | 0.50    |
| Cross Validation    | 0.39    |

---

## 📈 Analisis

* Model menunjukkan **overfitting**, ditandai dengan akurasi training yang sangat tinggi (100%) namun menurun pada data uji.
* Penambahan fitur **Hu Moments** meningkatkan performa dibanding hanya menggunakan fitur dasar.
* Nilai akurasi masih terbatas karena:

  * jumlah dataset kecil
  * variasi bentuk objek terbatas
  * kompleksitas bentuk (earphone cukup sulit dikenali)

---

## 📌 Kesimpulan

* Shape analysis efektif untuk objek dengan perbedaan bentuk yang jelas
* Fitur invariant seperti Hu Moments sangat penting
* Dataset yang kecil menyebabkan performa model kurang stabil

---

## 🚀 Saran Pengembangan

* Menambah jumlah dataset
* Menggunakan objek dengan bentuk lebih kontras
* Menggabungkan fitur shape dengan tekstur atau warna
* Menggunakan model klasifikasi yang lebih kompleks

---

## 🧪 Teknologi yang Digunakan

* Python
* OpenCV
* NumPy
* scikit-learn

---

## 📂 Struktur Folder

```
dataset/
├── Dompet/
├── Earphone/
├── Charger/

AnalisisBentuk.py
```

---

## ▶️ Cara Menjalankan

1. Pastikan dataset sudah sesuai struktur folder
2. Install dependencies:

   ```bash
   pip install opencv-python numpy scikit-learn
   ```
3. Jalankan program:

   ```bash
   python AnalisisBentuk.py
   ```
