from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import jwt

from shared.config.settings import AssetType, SETTINGS

app = FastAPI(title="Quantum Trading API Gateway")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Pydantic Models
class SignalRequest(BaseModel):
    asset: str
    asset_type: AssetType
    lookback_days: int = 30

class SignalResponse(BaseModel):
    signal_id: str
    asset: str
    action: str
    entry_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    timestamp: datetime

class TrainingRequest(BaseModel):
    asset_type: AssetType
    assets: List[str]
    start_date: str
    end_date: str

# Authentication
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=SETTINGS.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SETTINGS.SECRET_KEY, algorithm=SETTINGS.ALGORITHM)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Implement actual authentication
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Signal endpoints
@app.post("/api/v1/signals/generate", response_model=SignalResponse)
async def generate_signal(
    request: SignalRequest,
    token: str = Depends(oauth2_scheme)
):
    """Generate trading signal for an asset"""
    # Route to appropriate asset service
    # This would call the respective microservice
    pass

@app.get("/api/v1/signals/history")
async def get_signal_history(
    asset: Optional[str] = None,
    limit: int = 100,
    token: str = Depends(oauth2_scheme)
):
    """Get historical signals"""
    pass

# Training endpoints
@app.post("/api/v1/train")
async def trigger_training(
    request: TrainingRequest,
    token: str = Depends(oauth2_scheme)
):
    """Trigger model training"""
    # Send to training worker via Celery
    pass

# Model management
@app.get("/api/v1/models")
async def list_models(token: str = Depends(oauth2_scheme)):
    """List all models in registry"""
    pass

# Monitoring
@app.get("/api/v1/drift/status")
async def get_drift_status(
    asset: str,
    token: str = Depends(oauth2_scheme)
):
    """Get drift detection status"""
    pass