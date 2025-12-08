from evidently import Report
from evidently.presets import DataDriftPreset
from datetime import datetime

from config.constants import CONSTANTS
from database.db import SessionLocal
from database.models import Asset, DriftAlert
from services.feature_store import FeatureStore
from utilities.logger import logger


class MonitoringService:
    def __init__(self, asset_symbol: str):
        self.asset = asset_symbol
        self.feature_store = FeatureStore()

    def check_data_drift(self):
        """
        Compare recent data (inference) with historical data (reference/training)
        to detect drift.
        """
        logger.info(f"Running drift check for {self.asset}...")

        # 1. Fetch Reference Data (e.g., last 6 months used for training)
        reference_data = self.feature_store.get_training_features(self.asset, period="6mo")

        # 2. Fetch Current Data (e.g., last 1 week)
        current_data = self.feature_store.get_training_features(self.asset, period="1wk")

        if reference_data.empty or current_data.empty:
            logger.warning("Insufficient data for drift check.")
            return

        # Select features to monitor
        features = CONSTANTS.FEATURE_COLS

        # 3. Generate Report
        report = Report(metrics=[
            DataDriftPreset(columns=features)
        ])

        report.run(reference_data=reference_data, current_data=current_data)

        # 4. Analyze Results
        results = report.as_dict()

        # In Evidently 0.7.x, the structure is slightly different depending on preset
        # We look for 'dataset_drift' metric result
        drift_metric = next((m for m in results['metrics'] if m['metric'] == 'DataDriftPreset'), None)

        drift_detected = False
        drift_share = 0.0

        if drift_metric:
            drift_detected = drift_metric['result']['dataset_drift']
            drift_share = drift_metric['result']['drift_share']
        else:
            # Fallback if preset structure differs, look for base metrics
            pass

        # 5. Log to DB
        self._log_alert_to_db(drift_detected, drift_share, results)

        if drift_detected:
            logger.warning(f"DATA DRIFT DETECTED for {self.asset} (Share: {drift_share:.2f})")
        else:
            logger.info(f"No data drift detected for {self.asset}.")

        return results

    def _log_alert_to_db(self, drift_detected: bool, drift_score: float, report_json: dict):
        db = SessionLocal()
        try:
            asset = db.query(Asset).filter(Asset.symbol == self.asset).first()
            if not asset:
                logger.warning(f"Asset {self.asset} not found in DB, skipping alert log.")
                return

            alert = DriftAlert(
                asset_id=asset.id,
                drift_detected=1 if drift_detected else 0,
                drift_score=drift_score,
                report_json=report_json,
                timestamp=datetime.now()
            )
            db.add(alert)
            db.commit()
            logger.info(f"Drift check result saved to DB for {self.asset}")
        except Exception as e:
            logger.error(f"Failed to save drift alert to DB: {e}")
        finally:
            db.close()
