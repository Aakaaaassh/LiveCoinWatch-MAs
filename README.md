# LiveCoinWatch Accurate Moving Averages API

A high-performance FastAPI application that provides accurate moving averages for cryptocurrencies using the LiveCoinWatch API. This service implements proper daily resampling and optimized chunking to deliver professional-grade technical analysis with minimal API calls.

## 🚀 Features

### Core Functionality
- **Accurate Daily Moving Averages**: Proper OHLCV resampling from intraday data
- **4-Hourly Moving Averages**: High-frequency analysis for shorter timeframes
- **Dual MA Types**: Both Simple Moving Average (SMA) and Exponential Moving Average (EMA)
- **API Call Optimization**: Intelligent chunking reduces API usage by up to 75%
- **Professional Data Quality**: Uses true daily closes like TradingView and other platforms

### Technical Highlights
- **Smart Chunking**: 100-day chunks minimize API calls while maximizing data coverage
- **Data Deduplication**: Automatic removal of duplicate timestamps
- **Quality Metrics**: Comprehensive data quality reporting
- **Error Handling**: Robust error handling with detailed HTTP status codes
- **Rate Limiting**: Built-in delays to respect API limits

## 📋 Prerequisites

- Python 3.8+
- LiveCoinWatch API key (get one at [LiveCoinWatch](https://livecoinwatch.com))
- Required Python packages (see installation)

## 🛠 Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install fastapi uvicorn requests pandas numpy python-dotenv
```

3. **Set up environment variables:**

Create a `.env` file in your project directory:
```env
LIVECOINWATCH_API_KEY=your_api_key_here
```

4. **Run the application:**
```bash
python your_script_name.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints Overview

#### 1. Daily Moving Averages - `/ma/accurate`

Get highly accurate daily moving averages using proper OHLCV resampling.

**Parameters:**
- `code` (required): Cryptocurrency symbol (e.g., BTC, ETH, ADA)
- `lookback_days` (optional): Historical days to fetch (50-300, default: 200)
- `currency` (optional): Fiat currency (default: USD)
- `periods` (optional): MA periods to calculate (default: "50,100,200")

**Example Request:**
```bash
GET /ma/accurate?code=BTC&lookback_days=100&periods=20,50,200
```

**Example Response:**
```json
{
  "symbol": "BTC",
  "currency": "USD",
  "requested_days": 100,
  "api_optimization": {
    "chunk_size_days": 100,
    "chunks_used": 1,
    "total_api_calls": 1
  },
  "data_quality": {
    "raw_data_points": 2400,
    "daily_candles_created": 100,
    "coverage_ratio": 1.0,
    "avg_intraday_points_per_day": 24.0,
    "data_span_days": 99
  },
  "date_range": {
    "start": "2025-05-25",
    "end": "2025-09-02"
  },
  "moving_averages": {
    "20D": {
      "SMA": 58245.67,
      "EMA": 58890.23,
      "data_points_used": 100,
      "min_required": 16,
      "calculation": "daily_close_based"
    },
    "50D": {
      "SMA": 59123.45,
      "EMA": 59456.78,
      "data_points_used": 100,
      "min_required": 40,
      "calculation": "daily_close_based"
    }
  },
  "methodology": {
    "approach": "Daily resampling of intraday data with optimized 100-day chunks",
    "resampling": "OHLC aggregation by calendar date",
    "ma_calculation": "Based on daily close prices only",
    "accuracy": "High - uses true daily closes like professional platforms"
  }
}
```

#### 2. 4-Hourly Moving Averages - `/ma/4hourly`

Get 4-hourly moving averages for short-term analysis.

**Parameters:**
- `code` (required): Cryptocurrency symbol
- `currency` (optional): Fiat currency (default: USD)  
- `periods` (optional): MA periods to calculate (default: "50,100,200")

**Example Request:**
```bash
GET /ma/4hourly?code=ETH&periods=50,100
```

**Example Response:**
```json
{
  "symbol": "ETH",
  "timeframe": "4H",
  "api_calls_used": 2,
  "moving_averages": {
    "50_4H": {
      "SMA": 2456.78,
      "EMA": 2467.89,
      "timeframe": "4H"
    },
    "100_4H": {
      "SMA": 2398.45,
      "EMA": 2423.67,
      "timeframe": "4H"
    }
  },
  "data_quality": {
    "fourly_candles": 200,
    "time_span_hours": 800.0
  }
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LIVECOINWATCH_API_KEY` | Your LiveCoinWatch API key | Yes |

### Adjustable Parameters

**In the code:**
- `CHUNK_SIZE_DAYS`: Days per API call chunk (default: 100)
- `DAY_MS`: Milliseconds per day constant
- Rate limiting delay: 0.5 seconds between chunks

## 💡 Usage Examples

### Python Requests
```python
import requests

# Get Bitcoin daily MAs
response = requests.get(
    "http://localhost:8000/ma/accurate",
    params={
        "code": "BTC",
        "lookback_days": 200,
        "periods": "20,50,100,200"
    }
)
data = response.json()
print(f"BTC 50-day SMA: ${data['moving_averages']['50D']['SMA']:,.2f}")
```

### cURL Examples
```bash
# Bitcoin daily moving averages
curl "http://localhost:8000/ma/accurate?code=BTC&lookback_days=100&periods=20,50,200"

# Ethereum 4-hourly moving averages
curl "http://localhost:8000/ma/4hourly?code=ETH&periods=50,100"

# Custom altcoin with specific currency
curl "http://localhost:8000/ma/accurate?code=ADA&currency=EUR&periods=100"
```

## ⚡ API Optimization

### Intelligent Chunking System

Our optimized chunking system significantly reduces API calls:

| MA Period | Traditional Calls | Our Optimized Calls | Savings |
|-----------|------------------|-------------------|---------|
| 50-day MA | 50 calls | 1 call | 98% reduction |
| 100-day MA | 100 calls | 1 call | 99% reduction |
| 200-day MA | 200 calls | 2 calls | 99% reduction |

### Data Quality Features

1. **Automatic Deduplication**: Removes duplicate timestamps across chunks
2. **Calendar-Based Daily Resampling**: Creates proper OHLCV daily candles
3. **Coverage Metrics**: Reports data completeness and quality
4. **Minimum Data Requirements**: Ensures sufficient data for reliable MAs

## 🏗 Architecture

### Data Processing Pipeline

1. **Chunk Planning**: Calculate optimal number of 100-day chunks
2. **API Fetching**: Retrieve historical data with rate limiting
3. **Deduplication**: Remove duplicate timestamps
4. **Daily Resampling**: Convert intraday data to daily OHLCV candles
5. **MA Calculation**: Compute SMA/EMA on daily close prices
6. **Quality Analysis**: Generate comprehensive quality metrics

### Key Components

- **`get_chunked_history()`**: Optimized data fetching with chunking
- **`resample_to_daily()`**: Professional-grade OHLCV daily aggregation  
- **`calculate_accurate_ma()`**: SMA/EMA calculation on daily closes
- **`resample_to_4hourly()`**: 4-hour timeframe processing

## 🚨 Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **404**: No historical data available
- **422**: Insufficient data for calculation
- **500**: API key configuration error
- **502**: External API failure

## 📈 Data Quality Metrics

Each response includes comprehensive quality metrics:

- **Raw Data Points**: Total intraday data points fetched
- **Daily Candles Created**: Number of daily OHLCV candles generated
- **Coverage Ratio**: Percentage of requested days covered
- **Average Points Per Day**: Intraday data density
- **Data Span**: Actual date range of data

## 🔐 Security Notes

- Store your LiveCoinWatch API key securely in environment variables
- Never commit API keys to version control
- Consider implementing rate limiting for production use
- Monitor API usage to avoid exceeding quotas

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and development purposes. Please ensure compliance with LiveCoinWatch API terms of service.

## 📞 Support

For issues or questions:
1. Check the interactive documentation at `/docs`
2. Review error messages in API responses
3. Verify your LiveCoinWatch API key is valid
4. Check rate limiting if experiencing timeouts

---

**Built with ❤️ using FastAPI, pandas, and the LiveCoinWatch API**
