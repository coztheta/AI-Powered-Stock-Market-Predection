# 📈 AI Trading Platform

### Explore stock market data and experiment with hybrid stock-price forecasting

A Python and Flask-based stock market analysis and prediction platform that fetches historical market data using Yahoo Finance and uses a hybrid LSTM–ARIMA model to generate experimental stock-price forecasts.

**Project Status:** In Progress

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Application-black?logo=flask)
![Machine Learning](https://img.shields.io/badge/ML-LSTM%20%2B%20ARIMA-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

---

## ✨ Project Overview

The AI Trading Platform is a web-based stock market analysis and forecasting application developed using Python, Flask, and machine learning.

The application retrieves historical stock market data, provides an interactive dashboard, and combines LSTM neural networks with ARIMA time-series forecasting to experiment with future stock-price predictions.

The project is also being expanded to incorporate data engineering concepts such as automated ETL processing, structured data storage, data validation, and SQL-based analytics.

## 🚀 Features Implemented

- **Stock Market Dashboard:** Interactive dashboard with a searchable stock universe covering Indian and US equities and ETFs.
- **Historical Stock Data:** Retrieves historical market data using Yahoo Finance.
- **Hybrid Forecasting:** Combines LSTM and ARIMA models for stock-price forecasting experiments.
- **Model Training:** Trains machine learning models using historical stock data.
- **Model Persistence:** Saves trained LSTM models, fitted scalers, and ARIMA model artifacts for reuse.
- **User Authentication:** Login and signup functionality for local application demonstrations.
- **REST API Endpoints:** Provides JSON endpoints for stock listings, stock information, historical data, training, and predictions.

## 🧰 Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Framework | Flask, Flask-CORS, Jinja |
| Data Analysis | Pandas, NumPy |
| Data Source | Yahoo Finance, yfinance |
| Machine Learning | TensorFlow, Keras, Scikit-learn |
| Time-Series Forecasting | LSTM, ARIMA, Statsmodels |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Model Persistence | Joblib, Keras model saving |
| Version Control | Git, GitHub |

## 🏗️ Application Workflow

1. Retrieve historical stock market data from Yahoo Finance.
2. Process and prepare the data for forecasting.
3. Train the LSTM and ARIMA models.
4. Combine model predictions using hybrid ensemble weights.
5. Save trained model artifacts locally.
6. Generate stock-price forecasts through Flask API endpoints.
7. Display stock information and prediction results through the web dashboard.

## 📊 Data Analysis and Engineering Roadmap

The following features are planned as part of the project's next development stages:

- [ ] Build a Python-based ETL pipeline using Pandas.
- [ ] Implement data cleaning, transformation, and validation.
- [ ] Integrate PostgreSQL or another SQL database for historical stock data.
- [ ] Develop SQL queries for daily returns, price changes, trading volume, and volatility.
- [ ] Implement incremental data loading to process newly available market data.
- [ ] Develop structured analytical datasets for dashboard reporting.
- [ ] Improve model evaluation using forecasting performance metrics.
- [ ] Implement loading of saved models after application restarts.
- [ ] Enhance dashboard analytics and visualizations.

## ⚙️ Installation and Setup

### Requirements

- Python 3
- Internet connection for Yahoo Finance data retrieval

### 1. Clone the repository

```bash
git clone https://github.com/coztheta/AI-Powered-Stock-Market-Predection.git
```

### 2. Navigate to the project directory

```bash
cd AI-Powered-Stock-Prediction-main
```

### 3. Create a virtual environment

```bash
py -m venv .venv
```

### 4. Activate the virtual environment (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow may need to be installed separately if it is not included in the project's requirements:

```bash
pip install tensorflow
```

Use a TensorFlow version compatible with your Python environment.

### 6. Run the application

```bash
python app.py
```

Open the application in your browser:

http://127.0.0.1:5000

Create a demo account to access the dashboard.

## 🔌 API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/stocks` | List supported stocks |
| GET | `/api/stock/<ticker>` | Retrieve stock information |
| GET | `/api/historical/<ticker>?period=1y` | Retrieve historical stock data |
| POST | `/api/train_hybrid/<ticker>` | Train the hybrid prediction model |
| GET | `/api/predict_hybrid/<ticker>?days=30` | Generate stock-price predictions |
| POST | `/api/auth/signup` | Create a demo account |
| POST | `/api/auth/login` | Log in |
| POST | `/api/auth/logout` | Log out |

Supported stock tickers are defined in `STOCK_UNIVERSE` in `app.py`. Prediction requests accept a `days` parameter between 1 and 90.

## 📁 Project Structure

```text
AI-Powered-Stock-Market-Prediction/
│
├── app.py
├── requirements.txt
├── models/
│
├── static/
│   ├── css/
│   │   └── trading_ui.css
│   └── js/
│       └── main.js
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   └── signup.html
│
├── .gitignore
└── README.md
```

## 🔐 Security and Development Notes

- Demo user accounts are stored in memory and are lost when the server restarts.
- Set a secure `FLASK_SECRET_KEY` environment variable before deployment.
- Do not expose Flask's development server directly to the public internet.
- Market data availability depends on Yahoo Finance.
- The quote endpoint may return clearly labeled simulated values when live data cannot be retrieved.
- Model artifacts, virtual environments, research PDFs, and local environment files are excluded from Git.

## 🎯 Project Objectives

This project is intended to strengthen practical knowledge and experience in:

- Python programming and data analysis.
- Machine learning and time-series forecasting.
- Data preprocessing and model training.
- Flask application development and REST APIs.
- ETL pipeline development and data engineering.
- SQL-based data analysis and database management.

## ⚠️ Disclaimer

This project is developed for educational and experimental purposes only. Predictions are not financial advice or investment recommendations. Forecasts are not guaranteed to reflect actual market movements.

## 👨‍💻 Author

**coztheta**

GitHub: https://github.com/coztheta
