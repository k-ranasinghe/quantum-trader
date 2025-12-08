from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from database.db import Base


def generate_uuid():
    return str(uuid.uuid4())

class Asset(Base):
    __tablename__ = "assets"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    symbol = Column(String, unique=True, index=True, nullable=False)
    timeframe = Column(String, default="1h")
    is_active = Column(Integer, default=1)  # 1 for active, 0 for inactive
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    models = relationship("Model", back_populates="asset")
    signals = relationship("Signal", back_populates="asset_rel")
    alerts = relationship("DriftAlert", back_populates="asset_rel")

class Model(Base):
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    asset_id = Column(String, ForeignKey("assets.id"))
    name = Column(String, index=True)
    version = Column(String)
    file_path = Column(String, nullable=False)
    metrics = Column(JSON)  # Store accuracy, loss, etc.
    parameters = Column(JSON) # Hyperparameters
    status = Column(String, default="ready") # training, ready, failed, archived
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    asset = relationship("Asset", back_populates="models")
    generated_signals = relationship("Signal", back_populates="model_rel")

class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    asset_id = Column(String, ForeignKey("assets.id"))
    model_id = Column(String, ForeignKey("models.id"))
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    action = Column(String)  # Long, Short
    entry_min = Column(Float)
    entry_max = Column(Float)
    stop_loss = Column(Float)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    leverage = Column(Integer)
    confidence = Column(Float)
    models_agreement = Column(Float)
    
    # Optional: full signal dump for auditing
    full_details = Column(JSON)
    
    asset_rel = relationship("Asset", back_populates="signals")
    model_rel = relationship("Model", back_populates="generated_signals")

class DriftAlert(Base):
    __tablename__ = "drift_alerts"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    asset_id = Column(String, ForeignKey("assets.id"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    drift_score = Column(Float)
    drift_detected = Column(Integer) # 1 or 0
    report_json = Column(JSON) # Store full report result
    
    asset_rel = relationship("Asset", back_populates="alerts")
