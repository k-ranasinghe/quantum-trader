from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from shared.config.settings import AssetType

Base = declarative_base()


class SignalAction(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    signal_id = Column(String, unique=True, index=True)
    asset = Column(String, index=True)
    asset_type = Column(SQLEnum(AssetType))
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(SQLEnum(SignalAction))
    entry_price = Column(Float)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    position_size = Column(Float)
    confidence = Column(Float)
    models_agreement = Column(Float)
    regime = Column(String)
    expected_hold_days = Column(Integer)

    # Validation fields
    exit_price = Column(Float, nullable=True)
    exit_timestamp = Column(DateTime, nullable=True)
    realized_pnl = Column(Float, nullable=True)
    success = Column(Boolean, nullable=True)

    # Metadata
    model_versions = Column(JSON)
    features = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True)
    model_id = Column(String, unique=True, index=True)
    model_name = Column(String)
    model_type = Column(String)  # "ppo", "transformer", "hierarchical_rl"
    asset_type = Column(SQLEnum(AssetType))
    version = Column(String)
    status = Column(SQLEnum(ModelStatus))

    # Performance metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)

    # Paths
    model_path = Column(String)
    mlflow_run_id = Column(String)

    # Metadata
    hyperparameters = Column(JSON)
    training_data_period = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    deployed_at = Column(DateTime, nullable=True)


class DriftMetrics(Base):
    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True)
    asset = Column(String, index=True)
    asset_type = Column(SQLEnum(AssetType))
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Drift scores
    feature_drift_psi = Column(Float)
    prediction_drift_psi = Column(Float)
    concept_drift_score = Column(Float)

    # Details
    drift_detected = Column(Boolean)
    drifted_features = Column(JSON)

    # Actions
    retrain_triggered = Column(Boolean, default=False)