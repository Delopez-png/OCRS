Proyek "FastAPI OCR + Gemini + PostgreSQL" adalah sebuah sistem cerdas yang dirancang untuk mengubah gambar struk belanja menjadi data terstruktur yang akurat dan siap diolah. Tujuannya adalah untuk mengotomatisasi dan membersihkan data yang berasal dari dokumen tidak terstruktur.

Inti Kerja Sistem:
Pengenalan dan Ekstraksi Data (OCR dengan Gemini): Ketika gambar struk diunggah melalui FastAPI REST API, ia akan diproses oleh Google Gemini API. Gemini tidak hanya membaca teks (OCR), tetapi juga langsung mengekstrak informasi penting (nama produk, harga, kuantitas) dan menyusunnya menjadi format JSON terstruktur, menghilangkan kebutuhan akan pemrosesan teks mentah yang rumit.

Normalisasi dan Pembersihan Teks: Hasil teks dari Gemini sering kali mengandung kesalahan ejaan (typo) atau format yang tidak konsisten. Sistem menggunakan database PostgreSQL yang berisi data keyword normalisasi untuk secara otomatis memperbaiki dan menstandardisasi penulisan nama-nama produk.

Pencocokan Data (Fuzzy Matching): Setelah teks dinormalisasi, sistem menggunakan teknik Fuzzy Matching (dengan pustaka fuzzywuzzy) untuk mencocokkan nama produk dari struk dengan Data Master Produk yang tersimpan di PostgreSQL. Ini memastikan bahwa meskipun ada sedikit perbedaan ejaan, produk yang benar dari gudang data (misalnya data_pusat) tetap dapat diidentifikasi secara tepat.

Manajemen Data (PostgreSQL): Semua data—mulai dari data master, keyword normalisasi, hingga hasil akhir pemrosesan struk—disimpan dan dikelola secara terpusat di PostgreSQL, menjamin integritas dan memudahkan analisis data yang mendalam.

Kemudahan Penggunaan dan Integrasi: Proyek ini menawarkan REST API yang cepat dan andal, memungkinkan sistem lain (seperti aplikasi inventaris atau keuangan) untuk mengunggah struk dan menerima data bersih secara real-time. Proses deployment juga disederhanakan dengan skrip createTable.py yang otomatis menyiapkan seluruh database dan datanya

📦 project/
├── main.py # FastAPI utama (kode OCR, normalisasi, pencocokan)
├── createTable.py # Skrip pembuatan database & tabel
├── normalisasi.txt # Daftar keyword normalisasi awal
├── .env # File konfigurasi environment
├── requirements.txt # Daftar dependensi Python
├── SKU.xlsx # Data produk NIVEA
└── README.md # Dokumentasi proyek

### 1 Clone project tersebut
```bash
git clone https://github.com/Genco-png/OCR-SYSTEM.git
cd OCR-SYSTEM
```
### 2 Buat .venv
jika di windows:
```bash
python -m venv .venv
```
jika dilinux:
```bash
python3 -m venv .venv
```
### 3 Lalu Aktivkan .venv
jika di windows:
```bash
.venv/Scripts/activate
```
jika dilinux:
```bash
source venv/Scripts/activate
```
### 4 Isikan .env seperti .env.development
```bash
# PostgreSQL Database Config
DB_HOST=localhost
DB_PORT=5432
DB_DB=postgres
DB_USER=postgres
DB_PASSWORD=1221

# Google Generative AI
GOOGLE_API_KEY=AIzaSyDyeN-XMdmeGDltQByBiOnQJo4_jeKZ0mc

# Debug Mode
DEBUG=true
```
### 5 Jalankan Create Tabbel
```bash
python createTable.py
```

### 6 Jalankan OCRnya 
```bash 
python struk.py
```








