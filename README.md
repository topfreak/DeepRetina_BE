# Backend API - Deteksi Retinopati Diabetik

Backend REST API untuk sistem deteksi retinopati diabetik menggunakan deep learning. Aplikasi ini dibangun dengan FastAPI dan menggunakan model ensemble DenseNet121 + EfficientNetB0 untuk klasifikasi tingkat keparahan retinopati dari citra retina.

## 🚀 Fitur

- **Autentikasi & Autorisasi**: JWT-based authentication dengan bcrypt password hashing
- **Prediksi AI**: Klasifikasi retinopati diabetik dengan 5 tingkat keparahan (No DR, Mild, Moderate, Severe, Proliferative DR)
- **Riwayat Prediksi**: Penyimpanan dan pengambilan riwayat prediksi per user
- **Database**: SQLite dengan SQLAlchemy ORM
- **API Documentation**: Swagger UI & ReDoc auto-generated
- **CORS Support**: Configured untuk frontend integration
- **Containerization**: Docker support untuk deployment

## 📋 Tingkat Klasifikasi

| Kelas | Tingkat | Deskripsi |
|-------|---------|-----------|
| 0 | No DR | Tidak ada retinopati diabetik |
| 1 | Mild | Retinopati diabetik ringan |
| 2 | Moderate | Retinopati diabetik sedang |
| 3 | Severe | Retinopati diabetik parah |
| 4 | Proliferative DR | Retinopati diabetik proliferatif |

## 🛠️ Teknologi

- **Framework**: FastAPI
- **Database**: SQLite + SQLAlchemy
- **Authentication**: JWT + bcrypt
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: Pillow
- **Server**: Uvicorn ASGI
- **Validation**: Pydantic
- **Containerization**: Docker

## 📁 Struktur Project

```
backend/
├── main.py              # Aplikasi utama FastAPI
├── requirements.txt     # Dependencies Python
├── Dockerfile          # Container configuration
├── database.db         # SQLite database (auto-generated)
├── best_model_densenet_effnet.h5  # Model weights (diperlukan)
└── README.md           # Dokumentasi
```

## 🔧 Setup & Installation

### Prerequisites

- Python 3.11+
- pip
- Virtual environment (recommended)

### 1. Clone Repository

```bash
git clone https://github.com/topfreak/DeepRetina_BE.git
cd backend
```

### 2. Setup Virtual Environment

```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Model Weights

Pastikan file `best_model_densenet_effnet.h5` berada di root directory backend. Jika tidak ada, model akan menggunakan random weights.

### 5. Jalankan Aplikasi

```bash
# Development mode
python main.py

# Atau dengan uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🐳 Docker Deployment

### Method 1: Using Docker Compose (Recommended)

```bash
# Clone repository dan masuk ke direktori
git clone https://github.com/topfreak/DeepRetina_BE.git
cd backend

# Copy environment file dan sesuaikan
cp .env.example .env

# Buat direktori untuk model weights
mkdir -p models

# Download model weights dari Hugging Face
wget -O models/best_model_densenet_effnet.h5 https://huggingface.co/topikidx/deepretina/resolve/main/best_model_densenet_effnet.h5

# Jalankan dengan docker-compose
docker-compose up -d
```

### Method 2: Manual Docker Build

```bash
# Build image
docker build -t retinopathy-backend .

# Run container
docker run -d \
  --name retinopathy-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models:ro \
  -e SECRET_KEY="your_secret_key_here" \
  retinopathy-backend
```

### Method 3: Development with Docker

```bash
# Build development image
docker build -t retinopathy-backend:dev .

# Run in development mode
docker run -it \
  --name retinopathy-dev \
  -p 8000:8000 \
  -v $(pwd):/app \
  retinopathy-backend:dev
```

### Docker Commands

```bash
# Check container status
docker ps

# View logs
docker logs retinopathy-api

# Stop container
docker stop retinopathy-api

# Remove container
docker rm retinopathy-api

# Remove image
docker rmi retinopathy-backend
```

## 📖 API Documentation

Setelah menjalankan aplikasi, akses dokumentasi API di:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register` | Register user baru |
| POST | `/token` | Login dan mendapatkan access token |
| GET | `/users/me` | Get current user info |

### Prediction

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/predict` | Upload image dan prediksi | ✅ |
| GET | `/history` | Get riwayat prediksi user | ✅ |

### General

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |

## 📝 Usage Examples

### 1. Register User

```bash
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

### 2. Login

```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=password123"
```

### 3. Predict Image

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "file=@path/to/retina_image.jpg"
```

## 🔐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT secret key | `"rahasia_sangat_rahasia"` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Token expiration time | `60` |

## 📊 Database Schema

### Users Table
- `id`: Primary key
- `email`: Unique email address
- `hashed_password`: Bcrypt hashed password

### Prediction History Table
- `id`: Primary key
- `filename`: Uploaded image filename
- `predicted_class`: Predicted class (0-4)
- `confidence`: Model confidence score
- `timestamp`: Prediction timestamp
- `owner_id`: Foreign key to users table

## 🤖 Model Architecture

- **Ensemble Model**: DenseNet121 + EfficientNetB0
- **Input Shape**: (256, 256, 3)
- **Output**: 5 classes (softmax)
- **Training**: Fine-tuned with frozen early layers
- **Optimizer**: AdamW with weight decay

## 🚨 Error Handling

API mengembalikan error dalam format JSON:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid image, validation error)
- `401`: Unauthorized (invalid/expired token)
- `404`: Not Found
- `500`: Internal Server Error (model prediction failed)
- `503`: Service Unavailable (model not loaded)

## 📈 Performance

- **Image Processing**: Automatic resize to 256x256
- **Model Loading**: Loaded at startup for faster inference
- **Database**: SQLite untuk development, dapat diganti dengan PostgreSQL untuk production

## 🔮 Future Enhancements

- [ ] PostgreSQL support untuk production
- [ ] Redis caching untuk improved performance
- [ ] Batch prediction support
- [ ] Model versioning
- [ ] Advanced logging & monitoring
- [ ] Rate limiting
- [ ] Image preprocessing pipeline enhancement

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Taufiq Hidayatullah**
- Universitas: Universitas Amikom Yogyakarta
- Program Studi: S1-Informatika
- Email: topik.id.x@gmail.com

---

**Note**: 
- Pastikan file model weights `best_model_densenet_effnet.h5` tersedia untuk fungsi prediksi yang optimal.
- File model `.h5` dapat didownload dari Hugging Face: https://huggingface.co/topikidx/deepretina
- Setelah download, letakkan file model di root directory backend bersama dengan `main.py`