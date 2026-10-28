<div align="center">

# 📈 AI Trading Platform

### Explore market data and experiment with hybrid stock-price forecasts

Built with Flask, Yahoo Finance data, and an optional LSTM–ARIMA model.

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![ML](https://img.shields.io/badge/Models-LSTM%20%2B%20ARIMA-7B61FF)
![Status](https://img.shields.io/badge/Status-Academic%20Prototype-14B8A6)

</div>

---

## ✨ What it does

AI Trading Platform is a Flask-based stock-market dashboard and academic
prediction prototype. It combines a web interface with market data and an
optional hybrid forecasting workflow.

- **Market dashboard** with a searchable stock universe spanning Indian and
  US equities and ETFs.
- **Stock information and history** fetched from Yahoo Finance.
- **Hybrid forecasting experiment** that combines an LSTM neural network with
  an ARIMA model.
- **Account screens** for signing up and logging in during a local demo.
- **JSON endpoints** for stock listings, quotes, historical data, training,
  and predictions.

> [!WARNING]
> This project is for learning and experimentation only. Forecasts and displayed
> figures are not financial advice, investment recommendations, or guaranteed
> results. Market data may be delayed, unavailable, or simulated by the demo
> quote endpoint.

## 🧰 Built with

| Area | Technologies |
| --- | --- |
| Web application | Python, Flask, Flask-CORS, Jinja |
| Market data | `yfinance`, NumPy, pandas |
| Forecasting | TensorFlow/Keras (optional), scikit-learn, statsmodels |
| Interface | HTML, CSS, JavaScript, Chart.js |

## 🚀 Get started

### Requirements

- Python 3
- Internet access for Yahoo Finance market data

### Install

From the project directory, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow is optional and is not included in `requirements.txt`. Install a
TensorFlow build compatible with your Python version if you want to train the
LSTM component:

```powershell
pip install tensorflow
```

### Run

```powershell
python app.py
```

Open **http://localhost:5000** in your browser. The dashboard redirects to the
login page until you create a demo account.

## 🔌 API overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/stocks` | List supported stocks |
| `GET` | `/api/stock/<ticker>` | Get quote and company details |
| `GET` | `/api/historical/<ticker>?period=1y` | Get historical market data |
| `POST` | `/api/train_hybrid/<ticker>` | Train the hybrid model (requires TensorFlow) |
| `GET` | `/api/predict_hybrid/<ticker>?days=30` | Request a prediction |
| `POST` | `/api/auth/signup` | Create a demo account |
| `POST` | `/api/auth/login` | Sign in to a demo account |
| `POST` | `/api/auth/logout` | Sign out |

Supported tickers are defined in `STOCK_UNIVERSE` in `app.py`. Prediction
requests accept a `days` parameter from 1 to 90.

## 🗂️ Project layout

```text
.
├── app.py                 # Flask routes and hybrid forecasting logic
├── requirements.txt       # Core Python dependencies
├── models/                # Local model artifacts (not tracked)
├── static/
│   ├── css/trading_ui.css
│   └── js/main.js
└── templates/
    ├── base.html
    ├── index.html
    ├── login.html
    └── signup.html
```

## 🔐 Demo and deployment notes

- Demo users are held in memory and are lost when the server restarts. This is
  not a production authentication system.
- Set a strong `FLASK_SECRET_KEY` environment variable before deploying. When
  it is omitted, the app generates a temporary key for the current process.
- Do not expose Flask's development server directly to the public internet.
- The quote endpoint may return clearly labeled simulated values when live data
  cannot be retrieved. Historical data and model forecasts require live data.
- Research PDFs, model artifacts, virtual environments, and local environment
  files are intentionally excluded from Git.

## 📄 License

No license is currently included. All rights remain with their respective
authors until a license is added.
