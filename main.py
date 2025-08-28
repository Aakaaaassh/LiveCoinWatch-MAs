import os
import time
import math
from datetime import datetime, timezone, timedelta
import numpy as np
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
import dotenv

dotenv.load_dotenv()

LIVECOINWATCH_API = "https://api.livecoinwatch.com/coins/single/history"
API_KEY = os.getenv("LIVECOINWATCH_API_KEY")
DAY_MS = 24 * 60 * 60 * 1000
CHUNK_SIZE_DAYS = 50

app = FastAPI(title="LiveCoinWatch Accurate MA API", version="4.0")

def lcw_history_chunk(code: str, start_ms: int, end_ms: int, currency: str = "USD"):
    """Fetch a single chunk from LiveCoinWatch API"""
    headers = {
        "content-type": "application/json",
        "x-api-key": API_KEY
    }
    payload = {
        "currency": currency,
        "code": code.upper(),
        "start": start_ms,
        "end": end_ms,
        "meta": True
    }

    response = requests.post(LIVECOINWATCH_API, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        print(f"API Error: {response.status_code} - {response.text}")
        return []

    data = response.json()
    return data.get("history", [])

def get_chunked_history(code: str, target_days: int, currency: str = "USD"):
    """Get historical data using 50-day chunks"""
    chunks_needed = math.ceil(target_days / CHUNK_SIZE_DAYS)
    all_history = []

    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"Fetching {chunks_needed} chunks for {target_days} days of {code}")

    for chunk_idx in range(chunks_needed):
        chunk_end = end_time - (chunk_idx * CHUNK_SIZE_DAYS * DAY_MS)
        chunk_start = chunk_end - (CHUNK_SIZE_DAYS * DAY_MS)

        # Adjust the last chunk
        if chunk_idx == chunks_needed - 1:
            remaining_days = target_days - (chunk_idx * CHUNK_SIZE_DAYS)
            chunk_start = chunk_end - (remaining_days * DAY_MS)

        try:
            chunk_data = lcw_history_chunk(code, chunk_start, chunk_end, currency)
            if chunk_data:
                all_history.extend(chunk_data)
                print(f"  Chunk {chunk_idx + 1}/{chunks_needed}: Got {len(chunk_data)} data points")

            if chunk_idx < chunks_needed - 1:
                time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"  Chunk {chunk_idx + 1} failed: {e}")
            continue

    # Remove duplicates and sort
    unique_history = {}
    for item in all_history:
        unique_history[item['date']] = item

    sorted_history = sorted(unique_history.values(), key=lambda x: x['date'])
    print(f"Total unique data points: {len(sorted_history)}")

    return sorted_history

def resample_to_daily(df: pd.DataFrame):
    """
    Convert intraday data to proper daily OHLCV data
    This is the key improvement - we create true daily candles
    """
    # Set datetime index
    df['datetime'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.set_index('datetime')

    # Group by calendar date and create daily OHLCV
    df['date_only'] = df.index.date

    daily_data = []
    for date, group in df.groupby('date_only'):
        if len(group) > 0:
            # Sort by time to ensure proper OHLC
            group_sorted = group.sort_index()

            daily_candle = {
                'date': date,
                'open': float(group_sorted['rate'].iloc[0]),      # First price of day
                'high': float(group_sorted['rate'].max()),        # Highest price of day  
                'low': float(group_sorted['rate'].min()),         # Lowest price of day
                'close': float(group_sorted['rate'].iloc[-1]),    # Last price of day
                'volume': float(group_sorted['volume'].sum()),     # Total volume
                'data_points': len(group_sorted)                  # How many intraday points
            }
            daily_data.append(daily_candle)

    # Convert to DataFrame and sort by date
    daily_df = pd.DataFrame(daily_data)
    daily_df = daily_df.sort_values('date').reset_index(drop=True)

    return daily_df

def calculate_accurate_ma(daily_df: pd.DataFrame, periods: list):
    """
    Calculate moving averages on true daily data
    This ensures we're using actual daily closes, not intraday points
    """
    if len(daily_df) == 0:
        return {}

    # Use daily close prices
    closes = pd.Series(daily_df['close'].values, index=daily_df['date'])

    results = {}
    data_days = len(daily_df)

    print(f"Calculating MA on {data_days} true daily candles")

    for period in periods:
        if data_days >= int(period * 0.8):  # Need at least 80% of the period
            # Calculate SMA - simple moving average of daily closes
            sma_values = closes.rolling(window=period, min_periods=int(period * 0.8)).mean()
            sma_current = float(sma_values.iloc[-1]) if not pd.isna(sma_values.iloc[-1]) else None

            # Calculate EMA - exponential moving average of daily closes  
            ema_values = closes.ewm(span=period, adjust=False, min_periods=int(period * 0.8)).mean()
            ema_current = float(ema_values.iloc[-1]) if not pd.isna(ema_values.iloc[-1]) else None

            results[f"{period}D"] = {
                "SMA": sma_current,
                "EMA": ema_current,
                "data_points_used": data_days,
                "min_required": int(period * 0.8),
                "calculation": "daily_close_based"
            }

            print(f"  {period}D SMA: {sma_current:.2f}, EMA: {ema_current:.2f} (used {data_days} daily closes)")
        else:
            print(f"  {period}D: Insufficient data ({data_days} days, need {int(period * 0.8)})")

    return results

@app.get("/ma/accurate")
def get_accurate_ma(
    code: str = Query(..., description="Crypto code like BTC, ETH, ADA"),
    lookback_days: int = Query(200, ge=50, le=300, description="Lookback window (50-300 days)"),
    currency: str = Query("USD", description="Fiat currency"),
    periods: str = Query("50,100,200", description="MA periods to calculate")
):
    """
    Get highly accurate moving averages using proper daily resampling
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="LIVECOINWATCH_API_KEY not set")

    # Parse requested periods
    period_list = [int(p.strip()) for p in periods.split(",") if p.strip().isdigit()]

    # Fetch historical data
    try:
        history = get_chunked_history(code, lookback_days, currency)
        if not history:
            raise HTTPException(status_code=404, detail="No historical data available")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch data: {str(e)}")

    # Convert to DataFrame
    df = pd.DataFrame(history)
    print(f"Raw data: {len(df)} intraday points over {lookback_days} days")

    # Resample to true daily data
    daily_df = resample_to_daily(df)
    print(f"Daily data: {len(daily_df)} daily candles")

    if len(daily_df) < 30:
        raise HTTPException(status_code=422, detail=f"Insufficient daily data: {len(daily_df)} days")

    # Calculate accurate moving averages
    ma_results = calculate_accurate_ma(daily_df, period_list)

    if not ma_results:
        raise HTTPException(status_code=422, detail="Could not calculate any moving averages")

    # Analyze data quality
    coverage_ratio = len(daily_df) / lookback_days
    avg_points_per_day = len(df) / len(daily_df) if len(daily_df) > 0 else 0

    return {
        "symbol": code.upper(),
        "currency": currency.upper(),
        "requested_days": lookback_days,
        "data_quality": {
            "raw_data_points": len(df),
            "daily_candles_created": len(daily_df),
            "coverage_ratio": round(coverage_ratio, 3),
            "avg_intraday_points_per_day": round(avg_points_per_day, 1),
            "data_span_days": (daily_df['date'].iloc[-1] - daily_df['date'].iloc[0]).days if len(daily_df) > 1 else 0
        },
        "date_range": {
            "start": str(daily_df['date'].iloc[0]) if len(daily_df) > 0 else None,
            "end": str(daily_df['date'].iloc[-1]) if len(daily_df) > 0 else None
        },
        "moving_averages": ma_results,
        "methodology": {
            "approach": "Daily resampling of intraday data",
            "resampling": "OHLC aggregation by calendar date",
            "ma_calculation": "Based on daily close prices only",
            "accuracy": "High - uses true daily closes like professional platforms"
        }
    }

'''
@app.get("/ma/debug")
def debug_data_quality(
    code: str = Query(..., description="Crypto code"),
    days: int = Query(7, ge=1, le=30, description="Days to analyze"),
    currency: str = Query("USD", description="Currency")
):
    """
    Debug endpoint to analyze raw data quality and resampling process
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set")

    # Get recent data
    history = get_chunked_history(code, days, currency)
    df = pd.DataFrame(history)

    # Show raw data sample
    df['datetime'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.sort_values('datetime')

    raw_sample = []
    for _, row in df.tail(20).iterrows():  # Last 20 points
        raw_sample.append({
            "datetime": row['datetime'].isoformat(),
            "rate": row['rate'],
            "volume": row['volume']
        })

    # Create daily data
    daily_df = resample_to_daily(df)

    daily_sample = []
    for _, row in daily_df.tail(10).iterrows():  # Last 10 days
        daily_sample.append({
            "date": str(row['date']),
            "open": row['open'],
            "high": row['high'], 
            "low": row['low'],
            "close": row['close'],
            "volume": row['volume'],
            "intraday_points": row['data_points']
        })

    return {
        "symbol": code.upper(),
        "analysis_days": days,
        "raw_data": {
            "total_points": len(df),
            "sample_recent_20": raw_sample
        },
        "daily_data": {
            "total_days": len(daily_df),
            "sample_recent_10": daily_sample
        },
        "intervals": {
            "raw_avg_hours": round((df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 3600 / len(df), 1) if len(df) > 1 else 0,
            "daily_coverage": round(len(daily_df) / days, 2)
        }
    }
    '''

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

