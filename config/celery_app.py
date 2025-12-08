from celery import Celery

from config.settings import settings


celery_app = Celery(
    "quant_trader",
    broker=settings.RABBITMQ_URL,
    backend=settings.REDIS_URL,
    include=['utilities.ml_tasks']
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Periodic tasks beat schedule
    beat_schedule={
        'drift-check-every-day': {
            'task': 'utilities.ml_tasks.run_drift_check',
            'schedule': 86400.0, # 1 day
        },
    }
)
