import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from main import app
from database.db import get_db, Base


# Setup in-memory DB for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def client():
    # Create tables
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    # Drop tables
    Base.metadata.drop_all(bind=engine)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_create_asset(client):
    response = client.post("/assets", json={"symbol": "BTC-USD", "timeframe": "1h"})
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "BTC-USD"
    assert "id" in data

def test_create_duplicate_asset(client):
    # Try creating the same asset again
    response = client.post("/assets", json={"symbol": "BTC-USD", "timeframe": "1h"})
    assert response.status_code == 400

def test_list_assets(client):
    response = client.get("/assets")
    assert response.status_code == 200
    assert len(response.json()) == 1

def test_get_signals_empty(client):
    response = client.get("/signals")
    assert response.status_code == 200
    assert response.json() == []
