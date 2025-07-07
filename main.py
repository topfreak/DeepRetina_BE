# --- 1. Imports ---
import os
import io
import datetime
from typing import List, Optional

# FastAPI & Related
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware

# Database (SQLAlchemy & SQLite)
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# Pydantic (Data Validation)
from pydantic import BaseModel

# Password & Token Security
from passlib.context import CryptContext
from jose import JWTError, jwt

# ML & Image Processing
import numpy as np
import tensorflow as tf
from PIL import Image
# SOLUSI: Pindahkan import layer dan model ke sini untuk digunakan dalam pembuatan model
from tensorflow.keras.applications import DenseNet121, EfficientNetB0
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_densenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_effnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW


# --- 2. Konfigurasi ---
# Konfigurasi database SQLite
SQLALCHEMY_DATABASE_URL = "sqlite:///./database.db"

# Konfigurasi keamanan JWT
SECRET_KEY = os.getenv("SECRET_KEY", "rahasia_sangat_rahasia")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # Token berlaku selama 60 menit

# SOLUSI: Ganti path ke file bobot (.h5)
MODEL_WEIGHTS_PATH = "best_model_densenet_effnet.h5"

# --- 3. Setup Database ---
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 4. Model Database (SQLAlchemy ORM) ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    predictions = relationship("PredictionHistory", back_populates="owner")

class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    predicted_class = Column(Integer)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="predictions")

# Buat tabel di database jika belum ada
Base.metadata.create_all(bind=engine)

# --- 5. Skema Data (Pydantic) ---
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    class Config:
        from_attributes = True

class PredictionResult(BaseModel):
    filename: str
    predicted_class: int
    confidence: float
    message: str

class HistoryItem(BaseModel):
    filename: str
    predicted_class: int
    confidence: float
    timestamp: datetime.datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# --- 6. Utilitas Keamanan & Autentikasi ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    user = get_user(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

# --- 7. Inisialisasi Aplikasi FastAPI & Pemuatan Model ---
app = FastAPI(
    title="API Deteksi Retinopati Diabetik",
    description="Backend untuk klasifikasi tingkat keparahan retinopati dari citra retina.",
    version="1.0.0"
)


# --- SOLUSI DEPLOYMENT: KONFIGURASI CORS ---
# Definisikan domain frontend yang akan Anda gunakan nanti
origins = [
    "http://localhost",
    "http://localhost:8000", 
    "https://nama-app-frontend.vercel.app",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variabel global untuk menyimpan model yang sudah dimuat
model = None

# SOLUSI: Fungsi untuk membuat arsitektur model dari awal
def create_model():
    """Membangun arsitektur model Keras sesuai dengan notebook."""
    input_layer = Input(shape=(256, 256, 3), name='input_image')

    # Preprocessing untuk masing-masing backbone
    densenet_input = Lambda(preprocess_densenet)(input_layer)
    effnet_input = Lambda(preprocess_effnet)(input_layer)

    # DenseNet121
    densenet_base = DenseNet121(include_top=False, weights='imagenet', input_tensor=densenet_input)
    for layer in densenet_base.layers[:60]:
        layer.trainable = False
    densenet_out = GlobalAveragePooling2D()(densenet_base.output)

    # EfficientNetB0
    effnet_base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=effnet_input)
    for layer in effnet_base.layers[:60]:
        layer.trainable = False
    effnet_out = GlobalAveragePooling2D()(effnet_base.output)

    # Gabungkan fitur dari kedua backbone
    merged = Concatenate()([densenet_out, effnet_out])

    # Fully connected layers
    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer
    output = Dense(5, activation='softmax')(x)
    
    # Buat dan compile model
    created_model = Model(inputs=input_layer, outputs=output)
    created_model.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return created_model

@app.on_event("startup")
async def startup_event():
    """
    Event startup: Membangun arsitektur dan memuat bobot model.
    """
    global model
    print("Building model architecture...")
    model = create_model()
    print("Model architecture created.")
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Loading model weights from: {MODEL_WEIGHTS_PATH}")
        model.load_weights(MODEL_WEIGHTS_PATH)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Model weights file not found at {MODEL_WEIGHTS_PATH}. The model will have random weights.")


# --- 8. Endpoint API ---

@app.get("/", tags=["General"])
async def root():
    return {"message": "Selamat datang di API Deteksi Retinopati Diabetik"}

@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=UserInDB, tags=["Authentication"])
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/me", response_model=UserInDB, tags=["Users"])
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_retinopathy(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
        
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    if predicted_class_index < len(class_names):
        predicted_label = class_names[predicted_class_index]
        message = f"Gambar terdeteksi sebagai: {predicted_label} (Tingkat {predicted_class_index})"
    else:
        predicted_label = "Unknown"
        message = "Kelas yang diprediksi tidak dikenal."

    history = PredictionHistory(
        filename=file.filename,
        predicted_class=int(predicted_class_index),
        confidence=confidence,
        owner_id=current_user.id
    )
    db.add(history)
    db.commit()
    
    return {
        "filename": file.filename,
        "predicted_class": int(predicted_class_index),
        "confidence": confidence,
        "message": message
    }

@app.get("/history", response_model=List[HistoryItem], tags=["Prediction"])
async def get_prediction_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    history = db.query(PredictionHistory).filter(PredictionHistory.owner_id == current_user.id).all()
    return history

# --- 9. Menjalankan Aplikasi (jika file ini dieksekusi langsung) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
