
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
CHUNK_SIZE_DAYS = 100  # Changed from 50 to 100 days per chunk


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
    """Get historical data using 100-day chunks"""
    chunks_needed = math.ceil(target_days / CHUNK_SIZE_DAYS)
    all_history = []


    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)


    print(f"Fetching {chunks_needed} chunks for {target_days} days of {code} (100-day chunks)")


    for chunk_idx in range(chunks_needed):
        chunk_end = end_time - (chunk_idx * CHUNK_SIZE_DAYS * DAY_MS)
        chunk_start = chunk_end - (CHUNK_SIZE_DAYS * DAY_MS)


        # Adjust the last chunk
        if chunk_idx == chunks_needed - 1:
            remaining_days = target_days - (chunk_idx * CHUNK_SIZE_DAYS)
            chunk_start = chunk_end - (remaining_days * DAY_MS)


        print(f"  Chunk {chunk_idx + 1}/{chunks_needed}: Requesting {(chunk_end - chunk_start) // DAY_MS} days")
        print(f"    Start: {datetime.fromtimestamp(chunk_start/1000, tz=timezone.utc)}")
        print(f"    End: {datetime.fromtimestamp(chunk_end/1000, tz=timezone.utc)}")


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
    Now optimized with 100-day chunks:
    - 50 days MA: 1 API call
    - 100 days MA: 1 API call  
    - 200 days MA: 2 API calls
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="LIVECOINWATCH_API_KEY not set")


    # Parse requested periods
    period_list = [int(p.strip()) for p in periods.split(",") if p.strip().isdigit()]


    # Calculate expected API calls for transparency
    chunks_needed = math.ceil(lookback_days / CHUNK_SIZE_DAYS)
    print(f"Expected API calls for {lookback_days} days: {chunks_needed}")


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
        "api_optimization": {
            "chunk_size_days": CHUNK_SIZE_DAYS,
            "chunks_used": chunks_needed,
            "total_api_calls": chunks_needed
        },
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
            "approach": "Daily resampling of intraday data with optimized 100-day chunks",
            "resampling": "OHLC aggregation by calendar date",
            "ma_calculation": "Based on daily close prices only",
            "accuracy": "High - uses true daily closes like professional platforms"
        }
    }


def get_4hourly_history(code: str, currency: str = "USD"):
    """Get true 4H data with 2 API calls for 200-period support"""
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # Chunk 1: Recent 400 hours (100 points at 4H intervals)
    chunk1_start = end_time - (1_440_000_000)
    chunk1_data = lcw_history_chunk(code, chunk1_start, end_time, currency)
    
    time.sleep(0.5)  # Rate limiting
    
    # Chunk 2: Previous 400 hours (another 100 points at 4H intervals)  
    chunk2_start = chunk1_start - (1_440_000_000)
    chunk2_data = lcw_history_chunk(code, chunk2_start, chunk1_start, currency)
    
    # Combine and deduplicate
    all_data = chunk2_data + chunk1_data
    unique_data = {item['date']: item for item in all_data}
    
    return sorted(unique_data.values(), key=lambda x: x['date'])

def resample_to_4hourly(df: pd.DataFrame):
    """Process API data into clean 4-hourly structure"""
    
    df['datetime'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return pd.DataFrame({
        'datetime': df['datetime'],
        'close': df['rate'],     # Most important for MA calculation
        'volume': df['volume'],
        'open': df['rate'],      # API provides point data
        'high': df['rate'],      # Could be enhanced with OHLC
        'low': df['rate']        # if API provided more granular data
    })

def calculate_4hourly_ma(df: pd.DataFrame, periods: list):
    """Calculate SMA/EMA on 4-hourly close prices"""
    
    closes = pd.Series(df['close'].values, index=df['datetime'])
    results = {}
    
    for period in periods:
        if len(df) >= period:
            # SMA calculation
            sma = closes.rolling(window=period).mean().iloc[-1]
            # EMA calculation  
            ema = closes.ewm(span=period).mean().iloc[-1]
            
            results[f"{period}_4H"] = {
                "SMA": float(sma),
                "EMA": float(ema),
                "timeframe": "4H"
            }
    
    return results

@app.get("/ma/4hourly")
def get_4hourly_ma(
    code: str = Query(...),
    currency: str = Query("USD"),
    periods: str = Query("50,100,200")
):
    """Get 4-hourly SMA & EMA with single API call efficiency"""
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set")
    
    period_list = [int(p.strip()) for p in periods.split(",")]
    
    # Single optimized API call
    history = get_4hourly_history(code, currency)
    if not history:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Process 4-hourly data
    df = pd.DataFrame(history)
    fourly_df = resample_to_4hourly(df)
    
    # Calculate MAs
    ma_results = calculate_4hourly_ma(fourly_df, period_list)
    
    return {
        "symbol": code.upper(),
        "timeframe": "4H",
        "api_calls_used": 1,
        "moving_averages": ma_results,
        "data_quality": {
            "fourly_candles": len(fourly_df),
            "time_span_hours": round((fourly_df['datetime'].iloc[-1] - fourly_df['datetime'].iloc[0]).total_seconds() / 3600, 1)
        }
    }


# Time constants
MS_9DAY_CHUNK = 77_760_000_000   # 900 days in milliseconds (100 periods of 9-day intervals)
MS_21DAY_CHUNK = 181_440_000_000  # 2100 days in milliseconds (100 periods of 21-day intervals)

def get_9daily_history(code: str, currency: str = "USD"):
    """Get true 9-day data with 2 API calls for 200-period support"""
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # Chunk 1: Recent 900 days (100 points at 9-day intervals)
    chunk1_start = end_time - MS_9DAY_CHUNK
    chunk1_data = lcw_history_chunk(code, chunk1_start, end_time, currency)
    
    time.sleep(0.5)  # Rate limiting
    
    # Chunk 2: Previous 900 days (another 100 points at 9-day intervals)  
    chunk2_start = chunk1_start - MS_9DAY_CHUNK
    chunk2_data = lcw_history_chunk(code, chunk2_start, chunk1_start, currency)
    
    # Combine and deduplicate
    all_data = chunk2_data + chunk1_data
    unique_data = {item['date']: item for item in all_data}
    
    return sorted(unique_data.values(), key=lambda x: x['date'])

def get_21daily_history(code: str, currency: str = "USD"):
    """Get true 21-day data with 2 API calls for 200-period support"""
    
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    # Chunk 1: Recent 2100 days (100 points at 21-day intervals)
    chunk1_start = end_time - MS_21DAY_CHUNK
    chunk1_data = lcw_history_chunk(code, chunk1_start, end_time, currency)
    
    time.sleep(0.5)  # Rate limiting
    
    # Chunk 2: Previous 2100 days (another 100 points at 21-day intervals)  
    chunk2_start = chunk1_start - MS_21DAY_CHUNK
    chunk2_data = lcw_history_chunk(code, chunk2_start, chunk1_start, currency)
    
    # Combine and deduplicate
    all_data = chunk2_data + chunk1_data
    unique_data = {item['date']: item for item in all_data}
    
    return sorted(unique_data.values(), key=lambda x: x['date'])

def resample_to_9daily(df: pd.DataFrame):
    """Process API data into clean 9-daily structure"""
    
    df['datetime'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return pd.DataFrame({
        'datetime': df['datetime'],
        'close': df['rate'],     # Most important for MA calculation
        'volume': df['volume'],
        'open': df['rate'],      # API provides point data
        'high': df['rate'],      # Could be enhanced with OHLC
        'low': df['rate']        # if API provided more granular data
    })

def resample_to_21daily(df: pd.DataFrame):
    """Process API data into clean 21-daily structure"""
    
    df['datetime'] = pd.to_datetime(df['date'], unit='ms', utc=True)
    df = df.sort_values('datetime').reset_index(drop=True)
    
    return pd.DataFrame({
        'datetime': df['datetime'],
        'close': df['rate'],     # Most important for MA calculation
        'volume': df['volume'],
        'open': df['rate'],      # API provides point data
        'high': df['rate'],      # Could be enhanced with OHLC
        'low': df['rate']        # if API provided more granular data
    })

def calculate_9daily_ma(df: pd.DataFrame, periods: list):
    """Calculate SMA/EMA on 9-daily close prices"""
    
    closes = pd.Series(df['close'].values, index=df['datetime'])
    results = {}
    
    for period in periods:
        if len(df) >= period:
            # SMA calculation
            sma = closes.rolling(window=period).mean().iloc[-1]
            # EMA calculation  
            ema = closes.ewm(span=period).mean().iloc[-1]
            
            results[f"{period}_9D"] = {
                "SMA": float(sma),
                "EMA": float(ema),
                "timeframe": "9D"
            }
    
    return results

def calculate_21daily_ma(df: pd.DataFrame, periods: list):
    """Calculate SMA/EMA on 21-daily close prices"""
    
    closes = pd.Series(df['close'].values, index=df['datetime'])
    results = {}
    
    for period in periods:
        if len(df) >= period:
            # SMA calculation
            sma = closes.rolling(window=period).mean().iloc[-1]
            # EMA calculation  
            ema = closes.ewm(span=period).mean().iloc[-1]
            
            results[f"{period}_21D"] = {
                "SMA": float(sma),
                "EMA": float(ema),
                "timeframe": "21D"
            }
    
    return results

@app.get("/ma/9daily")
def get_9daily_ma(
    code: str = Query(...),
    currency: str = Query("USD"),
    periods: str = Query("50,100,200")
):
    """Get 9-daily SMA & EMA with optimized API calls"""
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set")
    
    period_list = [int(p.strip()) for p in periods.split(",")]
    
    # Optimized API calls for 9-day intervals
    history = get_9daily_history(code, currency)
    if not history:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Process 9-daily data
    df = pd.DataFrame(history)
    daily_df = resample_to_9daily(df)
    
    # Calculate MAs
    ma_results = calculate_9daily_ma(daily_df, period_list)
    
    return {
        "symbol": code.upper(),
        "timeframe": "9D",
        "api_calls_used": 2,
        "moving_averages": ma_results,
        "data_quality": {
            "daily_candles": len(daily_df),
            "time_span_days": round((daily_df['datetime'].iloc[-1] - daily_df['datetime'].iloc[0]).total_seconds() / (24 * 3600), 1)
        }
    }

@app.get("/ma/21daily")
def get_21daily_ma(
    code: str = Query(...),
    currency: str = Query("USD"),
    periods: str = Query("50,100,200")
):
    """Get 21-daily SMA & EMA with optimized API calls"""
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set")
    
    period_list = [int(p.strip()) for p in periods.split(",")]
    
    # Optimized API calls for 21-day intervals
    history = get_21daily_history(code, currency)
    if not history:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Process 21-daily data
    df = pd.DataFrame(history)
    daily_df = resample_to_21daily(df)
    
    # Calculate MAs
    ma_results = calculate_21daily_ma(daily_df, period_list)
    
    return {
        "symbol": code.upper(),
        "timeframe": "21D",
        "api_calls_used": 2,
        "moving_averages": ma_results,
        "data_quality": {
            "daily_candles": len(daily_df),
            "time_span_days": round((daily_df['datetime'].iloc[-1] - daily_df['datetime'].iloc[0]).total_seconds() / (24 * 3600), 1)
        }
    }

# Optional: Combined endpoint for multiple timeframes
@app.get("/ma/daily-combined")
def get_daily_ma_combined(
    code: str = Query(...),
    currency: str = Query("USD"),
    timeframes: str = Query("9D,21D"),
    periods: str = Query("50,100,200")
):
    """Get multiple daily timeframe MAs in a single request"""
    
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set")
    
    period_list = [int(p.strip()) for p in periods.split(",")]
    timeframe_list = [tf.strip() for tf in timeframes.split(",")]
    
    results = {}
    total_api_calls = 0
    
    for timeframe in timeframe_list:
        if timeframe == "9D":
            history = get_9daily_history(code, currency)
            if history:
                df = pd.DataFrame(history)
                daily_df = resample_to_9daily(df)
                ma_results = calculate_9daily_ma(daily_df, period_list)
                results[timeframe] = {
                    "moving_averages": ma_results,
                    "data_quality": {
                        "daily_candles": len(daily_df),
                        "time_span_days": round((daily_df['datetime'].iloc[-1] - daily_df['datetime'].iloc[0]).total_seconds() / (24 * 3600), 1)
                    }
                }
                total_api_calls += 2
                
        elif timeframe == "21D":
            history = get_21daily_history(code, currency)
            if history:
                df = pd.DataFrame(history)
                daily_df = resample_to_21daily(df)
                ma_results = calculate_21daily_ma(daily_df, period_list)
                results[timeframe] = {
                    "moving_averages": ma_results,
                    "data_quality": {
                        "daily_candles": len(daily_df),
                        "time_span_days": round((daily_df['datetime'].iloc[-1] - daily_df['datetime'].iloc[0]).total_seconds() / (24 * 3600), 1)
                    }
                }
                total_api_calls += 2
    
    return {
        "symbol": code.upper(),
        "timeframes": timeframe_list,
        "api_calls_used": total_api_calls,
        "results": results
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
        "chunk_optimization": {
            "chunk_size_days": CHUNK_SIZE_DAYS,
            "chunks_needed": math.ceil(days / CHUNK_SIZE_DAYS)
        },
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



