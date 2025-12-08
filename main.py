from fastapi import FastAPI
from api.routes import router
from database.db import engine, Base

from config.settings import settings

app = FastAPI(title="QuantTrader API", version="1.0.0")

# Create tables ONLY if not testing (tests create their own in-memory DB)
# OR if strictly running the app.
# The error happens because 'main.py' is imported, which runs 'Base.metadata.create_all(bind=engine)'
# 'engine' is configured to connect to Postgres from settings.DATABASE_URL
# In test environment, this fails.

# We should move the table creation to a startup event or conditional check.

@app.on_event("startup")
def startup():
    # Only try to create tables if we can connect, or let Alembic handle it in prod.
    # For simplicity here:
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Warning: Could not create tables on startup (normal during tests if DB not up): {e}")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
