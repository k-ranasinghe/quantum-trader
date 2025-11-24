# Quantum Trader

Quantum Trader is a comprehensive, microservices-based platform for algorithmic trading of stocks and cryptocurrencies. It leverages cutting-edge machine learning and deep learning models to generate trading signals, manage portfolios, and execute trades in a simulated or live environment.

## Key Features

- **Microservices Architecture**: A scalable and resilient architecture with services for API gateway, stock and crypto trading, MLOps, and background workers.
- **Ensemble of ML/DL Models**: Utilizes a variety of models including PPO, SAC, Transformer, CNN-LSTM, and Hierarchical RL for robust signal generation.
- **Automated Training Pipelines**: Features automated training pipelines with hyperparameter optimization using Optuna and experiment tracking with MLflow.
- **Real-time Data Integration**: Integrates with various data sources like yfinance, ccxt, Alpaca, and Polygon to fetch real-time and historical market data.
- **Monitoring and Alerting**: Includes a monitoring stack with Prometheus and Grafana for real-time metrics and alerting.
- **Backtesting Engine**: A built-in backtesting engine to evaluate trading strategies on historical data.
- **Drift Detection**: Proactively monitors for data and model drift to ensure model performance.

## Architecture

The platform is designed as a distributed system of microservices communicating via a message broker (RabbitMQ) and a RESTful API.

- **API Gateway**: The single entry point for all client interactions.
- **Trading Services**: Separate services for stock and crypto trading, each with its own set of models.
- **MLOps Service**: Manages the lifecycle of machine learning models, including training, deployment, and monitoring.
- **Training Workers**: Asynchronous workers that handle the computationally intensive model training tasks.
- **Data Stores**: Utilizes PostgreSQL for structured data, InfluxDB for time-series market data, and Redis for caching.
- **Monitoring**: Prometheus for metrics collection and Grafana for visualization.

## Technology Stack

- **Backend**: Python, FastAPI, SQLAlchemy, Alembic
- **Machine Learning**: PyTorch, Stable-Baselines3, FinRL, Gymnasium, Transformers, Timm
- **MLOps**: MLflow, Optuna, Evidently
- **Data**: Pandas, Polars, NumPy, TA-Lib
- **Databases**: PostgreSQL, InfluxDB, Redis
- **Message Broker**: RabbitMQ
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Monitoring**: Prometheus, Grafana

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- PostgreSQL 15+
- An `.env` file with the required environment variables (see `.env.example`)

### Local Setup with Docker

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/k-ranasinghe/quantum-trader.git
    cd quantum-trader
    ```

2.  **Set up the environment:**
    Create a `.env` file from the example and fill in the required values:
    ```bash
    cp .env.example .env
    ```

3.  **Build and run the services:**
    ```bash
    docker-compose up --build
    ```
    This will start all the services, including the database, message broker, and application services.

4.  **Access the API:**
     - API Gateway: http://localhost:8000
     - MLflow: http://localhost:5000
     - Grafana: http://localhost:3000
     - RabbitMQ: http://localhost:15672

### Kubernetes Deployment

For a production-like environment, you can deploy the application to a Kubernetes cluster. The `/k8s` directory contains the necessary manifest files.

1.  **Create the namespace and secrets:**
    ```bash
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/secrets.yaml
    ```

2.  **Deploy the services:**
    ```bash
    kubectl apply -f k8s/
    ```

## Usage

### API Endpoints

The API Gateway (`http://localhost:8000`) provides the following key endpoints:

-   `POST /token`: Authenticate and receive a JWT token.
-   `POST /api/v1/signals/generate`: Generate a trading signal for a given asset.
-   `GET /api/v1/signals/history`: Get a history of trading signals.
-   `POST /api/v1/train`: Trigger the training of a new model.
-   `GET /api/v1/models`: List all available models.

### Scripts

The `/scripts` directory contains useful scripts for common tasks:

-   `train_models.py`: Manually trigger the training of models.
-   `backtest_strategy.py`: Run a backtest of a trading strategy.
-   `generate_signals.py`: Manually generate trading signals.

## Directory Structure

```
quant-trader/
├── .github/          # CI/CD workflows
├── docker/           # Dockerfiles for each service
├── k8s/              # Kubernetes manifests
├── monitoring/       # Prometheus and Grafana configurations
├── requirements/     # Python dependencies
├── scripts/          # Utility scripts
├── services/         # Microservices source code
│   ├── api_gateway/
│   ├── crypto_service/
│   ├── dataops_service/
│   ├── mlops_service/
│   ├── securities_service/
│   └── stock_service/
├── shared/           # Shared code (models, database, utils)
│   ├── backtest/
│   ├── config/
│   ├── database/
│   ├── environments/
│   ├── models/
│   └── utils/
├── workers/          # Background workers for training and inference
├── .env.example      # Example environment variables
├── docker-compose.yml # Docker Compose configuration
└── main.py           # Main entry point
```