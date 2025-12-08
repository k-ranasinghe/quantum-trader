import traceback
from datetime import datetime
import os
import shutil
import numpy as np

from config.celery_app import celery_app
from config.settings import settings
from database.db import SessionLocal
from database.models import Signal as SignalModel, Asset, Model
from services.train_service import TrainService
from services.inference_service import InferenceService
from services.monitoring_service import MonitoringService
from utilities.logger import logger


@celery_app.task(bind=True)
def train_model_task(self, asset_symbol: str, timeframe: str, period: str):
    """
    Async task to train models for a specific asset.
    """
    logger.info(f"Starting training task for {asset_symbol}")
    try:
        service = TrainService(asset=asset_symbol, timeframe=timeframe, period=period)
        success = service.run_full_training_pipeline()

        if success:
            logger.info(f"Training completed for {asset_symbol}")

            # Move artifacts to shared storage if they are not already there
            # TrainService currently saves to local dir (or MLFlow).
            # We need to ensure they are in settings.MODEL_REGISTRY_DIR for InferenceService to pick them up.
            # In the updated Plan, we mounted /app/results.

            # Simple sync logic for prototype:
            # TrainService saves "best_EnhancedModel1.pth" etc in CWD.
            # We move them to registry dir.

            artifacts = [
                "best_EnhancedModel1.pth",
                "best_EnhancedModel2.pth",
                "scaler.pkl",
                "regime_detector.json",
                "meta_model.json",
                "rl_agent.zip"
            ]

            # Determine destination
            # We might want asset-specific subfolders, but InferenceService looks in registry_path
            # Let's keep it flat or assume single asset focus for prototype simplicity,
            # OR refactor InferenceService to look in subfolders.
            # For robustness, let's copy to registry_root.

            dest_dir = settings.MODEL_REGISTRY_DIR
            os.makedirs(dest_dir, exist_ok=True)

            for art in artifacts:
                if os.path.exists(art):
                    shutil.copy2(art, os.path.join(dest_dir, art))
                    logger.info(f"Copied {art} to {dest_dir}")

            # Update DB status
            db = SessionLocal()
            try:
                asset = db.query(Asset).filter(Asset.symbol == asset_symbol).first()
                if not asset:
                    asset = Asset(symbol=asset_symbol)
                    db.add(asset)
                    db.commit()

                model_entry = Model(
                    asset_id=asset.id,
                    name=f"Ensemble_{asset_symbol}",
                    version=datetime.now().strftime("%Y%m%d%H%M"),
                    file_path=dest_dir,
                    status="ready"
                )
                db.add(model_entry)
                db.commit()
            finally:
                db.close()
            return f"Training successful for {asset_symbol}"
        else:
            return f"Training failed for {asset_symbol}"
    except Exception as e:
        logger.error(f"Training task failed: {traceback.format_exc()}")
        raise e


@celery_app.task(bind=True)
def generate_signal_task(self, asset_symbol: str):
    """
    Async task to generate inference signal.
    """
    logger.info(f"Starting signal generation for {asset_symbol}")
    db = SessionLocal()
    try:
        # Initialize service with shared registry path
        service = InferenceService(
            asset_symbol=asset_symbol,
            model_registry_path=settings.MODEL_REGISTRY_DIR
        )

        # Generate Signal (returns config.models.Signal DTO)
        signal_dto = service.generate_signal()

        # Map to DB Model
        # Find asset ID
        asset = db.query(Asset).filter(Asset.symbol == asset_symbol).first()
        if not asset:
            logger.warning(f"Asset {asset_symbol} not in DB, creating...")
            asset = Asset(symbol=asset_symbol)
            db.add(asset)
            db.commit()
            db.refresh(asset)

        def to_python(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Create Signal Model
        db_signal = SignalModel(
            asset_id=asset.id,
            action=signal_dto.action,
            entry_min=to_python(signal_dto.entry_min),
            entry_max=to_python(signal_dto.entry_max),
            stop_loss=to_python(signal_dto.stop_loss),
            take_profit_1=to_python(signal_dto.take_profit_1),
            take_profit_2=to_python(signal_dto.take_profit_2),
            take_profit_3=to_python(signal_dto.take_profit_3),
            leverage=int(signal_dto.leverage),
            confidence=float(signal_dto.confidence),
            models_agreement=float(signal_dto.models_agreement),
            timestamp=datetime.now()
        )

        db.add(db_signal)
        db.commit()
        db.refresh(db_signal)

        logger.info(f"Signal saved to DB with ID: {db_signal.id}")
        return f"Signal generated and saved: {db_signal.id}"

    except Exception as e:
        logger.error(f"Signal task failed: {traceback.format_exc()}")
        db.rollback()
        raise e
    finally:
        db.close()


@celery_app.task
def run_drift_check():
    """
    Periodic task to check for drift on all active assets.
    """
    db = SessionLocal()
    try:
        assets = db.query(Asset).filter(Asset.is_active == 1).all()
        for asset in assets:
            monitor = MonitoringService(asset.symbol)
            monitor.check_data_drift()
    finally:
        db.close()
