Sistem Monitoring dan Prediksi Cuaca Terintegrasi - AWS Maritim Pontianak
1. Deskripsi Proyek
Proyek ini adalah sebuah sistem terintegrasi yang dirancang untuk melakukan monitoring cuaca secara real-time dan memberikan prediksi berbasis deep learning. Sistem ini dibangun dengan arsitektur modern yang terdiri dari tiga komponen utama: layanan backend untuk akuisisi data, modul deep learning untuk peramalan, dan antarmuka frontend untuk visualisasi interaktif.

Sumber data utama sistem ini berasal dari endpoint API Automatic Weather Station (AWS) Maritim Pontianak: http://202.90.199.132/aws-new/data/station/latest/3000000011.

2. Arsitektur Sistem
Sistem ini dirancang dengan arsitektur microservices-oriented untuk memastikan skalabilitas, ketahanan, dan kemudahan pemeliharaan.

+---------------------------+        +--------------------------+        +-------------------------+
|                           |        |                          |        |                         |
|   Sumber Data Eksternal   |        |     Layanan Backend      |        |   Antarmuka Frontend    |
| (API AWS Maritim Pontianak)|<------>|   (Python - FastAPI)     |<------>|      (React.js)         |
|                           |        |                          |        |                         |
+-------------+-------------+        +------------+-------------+        +------------+------------+
              |                          |         ^                          ^
              | Data (JSON)              | Data    | Model inference          | Visualisasi & Interaksi
              v                          v         |                          | Pengguna
+-------------+-------------+        +---+---------+------------+        +---+---------------------+
|                           |        |                          |
|   Database Time-Series    |        |  Modul Prediksi (AI/ML)  |
|     (PostgreSQL)          |<------>| (Python - TensorFlow)    |
|                           |        |                          |
+---------------------------+        +--------------------------+

Alur Kerja:

Layanan Backend secara periodik (setiap menit) mengambil data dari API eksternal.

Data yang berhasil diambil disimpan ke dalam Database Time-Series untuk membangun catatan historis.

Modul Prediksi menggunakan data historis dari database untuk melatih model-model deep learning.

Antarmuka Frontend mengambil data terkini dan historis dari backend untuk ditampilkan dalam bentuk kartu statistik dan grafik interaktif.

Frontend juga mengirim permintaan ke backend untuk mendapatkan hasil prediksi dari model AI/ML, yang kemudian divisualisasikan kepada pengguna.

3. Tumpukan Teknologi (Technology Stack)
Pemilihan teknologi didasarkan pada performa, skalabilitas, dan ekosistem yang matang untuk setiap komponen.

Komponen

Teknologi

Alasan Pemilihan

🖥️ Frontend

React.js (dengan TypeScript), Tailwind CSS, D3.js/Recharts

React menawarkan UI yang reaktif dan berbasis komponen. TypeScript memastikan kode yang type-safe dan lebih mudah dikelola. Tailwind CSS mempercepat pengembangan UI yang modern dan responsif. D3.js/Recharts menyediakan kapabilitas visualisasi data yang kuat dan interaktif, ideal untuk grafik cuaca dan windrose.

⚙️ Backend

Python, FastAPI, PostgreSQL (dengan TimescaleDB), Celery & Redis

FastAPI adalah framework web modern berbasis Python yang sangat cepat dan efisien. PostgreSQL dengan ekstensi TimescaleDB dioptimalkan untuk menangani data deret waktu (time-series) dalam volume besar. Celery dan Redis digunakan untuk mengelola tugas terjadwal (pengambilan data) secara asinkron dan andal.

🧠 AI/ML

Python, TensorFlow (Keras), Scikit-learn, Pandas, Jupyter

Python adalah standar industri untuk data science. TensorFlow (Keras) menyediakan API level tinggi untuk membangun arsitektur deep learning kompleks seperti LSTM dan GRU. Scikit-learn dan Pandas adalah pustaka esensial untuk prapemrosesan data dan rekayasa fitur.

📦 Deployment

Docker, Docker Compose

Docker digunakan untuk mengemas setiap layanan ke dalam kontainer yang terisolasi, memastikan konsistensi lingkungan dari pengembangan hingga produksi. Docker Compose menyederhanakan proses menjalankan keseluruhan aplikasi multi-container secara lokal.

4. Fitur Utama
A. Dasbor Monitoring
Kartu Statistik Real-time: Menampilkan 10 parameter cuaca terkini secara jelas.

Galeri Grafik Historis: Visualisasi setiap parameter dalam grafik garis individual.

Filter Interaktif: Pengguna dapat memfilter data berdasarkan rentang tanggal dan waktu.

Visualisasi Khusus: Grafik windrose untuk visualisasi intuitif dari data arah dan kecepatan angin.

B. Modul Prediksi Deep Learning
Sistem ini mengimplementasikan empat model prediksi spesifik:

Prediksi Potensi Banjir Rob / Pasang Air Laut:

Target: WATER LEVEL (m)

Output: Nilai prediksi ketinggian air untuk 1, 3, dan 6 jam ke depan.

Prediksi Kondisi untuk Pelayaran & Nelayan:

Target: WINDSPEED (m/s) & WIND DIRECTION (°)

Output: Nilai prediksi kecepatan dan arah angin untuk beberapa jam ke depan.

Prediksi Potensi Hujan Lokal:

Target: RAINFALL (mm)

Output: Probabilitas (dalam %) terjadinya hujan (RAINFALL > 0) dalam 1 hingga 2 jam ke depan.

Prediksi Suhu dan Kelembaban Perkotaan:

Target: AIR TEMP (°C) & HUMIDITY (%RH)

Output: Nilai prediksi suhu dan kelembaban untuk 6 jam ke depan.

5. Instalasi dan Menjalankan Sistem
Pastikan Anda telah menginstal prasyarat berikut di sistem Anda:

Docker

Docker Compose

Git

Langkah-langkah Instalasi:
Clone Repositori

git clone [https://github.com/your-username/weather-prediction-system.git](https://github.com/your-username/weather-prediction-system.git)
cd weather-prediction-system

Konfigurasi Lingkungan
Buat file .env di direktori root proyek dengan menyalin dari env.example. Sesuaikan isinya jika diperlukan (misalnya, kredensial database).

cp env.example .env

Bangun dan Jalankan Layanan menggunakan Docker Compose
Perintah ini akan membangun image Docker untuk setiap layanan (frontend, backend, database) dan menjalankannya dalam kontainer.

docker-compose up --build -d

-d (detached mode) akan menjalankan kontainer di latar belakang.

Akses Aplikasi

Frontend/Dasbor: Buka browser dan akses http://localhost:3000

Backend API Docs: Akses http://localhost:8000/docs untuk melihat dokumentasi API interaktif (Swagger UI).

Melatih Model Deep Learning:
Proses pelatihan model tidak dijalankan secara otomatis oleh docker-compose up untuk memberikan kontrol lebih.

Masuk ke Kontainer Backend
Pastikan kontainer sedang berjalan, lalu eksekusi perintah berikut untuk mendapatkan akses shell ke dalam kontainer layanan backend.

docker-compose exec backend /bin/bash

Jalankan Skrip Pelatihan
Di dalam shell kontainer, jalankan skrip pelatihan. Skrip ini akan mengambil data dari database, melakukan prapemrosesan, melatih keempat model, dan menyimpan artefak model yang telah dilatih ke dalam direktori yang sesuai.

python -m app.ml.train_models

Proses ini mungkin memakan waktu cukup lama, tergantung pada volume data historis dan kompleksitas model.

6. Struktur Proyek
.
├── backend/                  # Direktori layanan Backend (FastAPI)
│   ├── app/
│   │   ├── api/              # Logika endpoint API
│   │   ├── core/             # Konfigurasi aplikasi
│   │   ├── crud/             # Operasi database (Create, Read, Update, Delete)
│   │   ├── db/               # Inisialisasi & sesi database
│   │   ├── ml/               # Skrip prediksi & pelatihan model
│   │   ├── models/           # Model data SQLAlchemy
│   │   ├── schemas/          # Skema data Pydantic
│   │   └── services/         # Logika pengumpul data
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # Direktori antarmuka Frontend (React)
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/       # Komponen UI (Grafik, Kartu, Navbar)
│   │   ├── pages/            # Halaman utama (Dasbor, Prediksi)
│   │   ├── services/         # Logika untuk memanggil API backend
│   │   └── App.tsx
│   ├── Dockerfile
│   └── package.json
├── data/                     # Volume untuk persistensi data PostgreSQL
├── .env                      # File konfigurasi lingkungan (dibuat dari .env.example)
├── .env.example              # Contoh file konfigurasi
├── docker-compose.yml        # Definisi layanan untuk Docker Compose
└── README.md                 # Dokumentasi ini
