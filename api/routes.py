from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from config.constants import CONSTANTS
from database.db import get_db
from database.models import Signal as SignalModel, Asset as AssetModel
from utilities.ml_tasks import train_model_task, generate_signal_task

router = APIRouter()

# Pydantic models for API
class AssetCreate(BaseModel):
    symbol: str
    timeframe: str = "1h"

class AssetResponse(BaseModel):
    id: str
    symbol: str
    is_active: int
    
    class Config:
        from_attributes = True

class TrainRequest(BaseModel):
    asset_symbol: str
    timeframe: str = CONSTANTS.TIMEFRAME
    period: str = CONSTANTS.PERIOD

class SignalResponse(BaseModel):
    id: str
    asset_id: str
    action: str
    entry_min: float
    entry_max: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    leverage: int
    timestamp: datetime
    confidence: float
    
    class Config:
        from_attributes = True

@router.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now()}

@router.post("/assets", response_model=AssetResponse)
def create_asset(asset: AssetCreate, db: Session = Depends(get_db)):
    db_asset = db.query(AssetModel).filter(AssetModel.symbol == asset.symbol).first()
    if db_asset:
        raise HTTPException(status_code=400, detail="Asset already exists")
    
    new_asset = AssetModel(symbol=asset.symbol, timeframe=asset.timeframe)
    db.add(new_asset)
    db.commit()
    db.refresh(new_asset)
    return new_asset

@router.get("/assets", response_model=List[AssetResponse])
def list_assets(db: Session = Depends(get_db)):
    return db.query(AssetModel).all()

@router.post("/train")
def trigger_training(request: TrainRequest):
    """
    Trigger async training for an asset.
    """
    task = train_model_task.delay(request.asset_symbol, request.timeframe, request.period)
    return {"message": "Training started", "task_id": str(task.id)}

@router.post("/inference")
def trigger_inference(asset_symbol: str):
    """
    Trigger async inference for an asset.
    """
    task = generate_signal_task.delay(asset_symbol)
    return {"message": "Inference started", "task_id": str(task.id)}

@router.get("/signals", response_model=List[SignalResponse])
def get_signals(asset_symbol: Optional[str] = None, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(SignalModel)
    if asset_symbol:
        asset = db.query(AssetModel).filter(AssetModel.symbol == asset_symbol).first()
        if asset:
            query = query.filter(SignalModel.asset_id == asset.id)
    
    return query.order_by(SignalModel.timestamp.desc()).limit(limit).all()
