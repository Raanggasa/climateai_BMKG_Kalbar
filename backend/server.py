from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from collections import defaultdict
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(
    title="Weather Monitoring & Prediction System",
    description="BMKG AWS Pontianak - Monitoring dan Prediksi Cuaca",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Weather API Configuration
WEATHER_API_URL = "http://202.90.199.132/aws-new/data/station/latest/3000000011"

# Define Models
class WeatherData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idaws: str
    waktu: datetime
    windspeed: float = Field(description="Wind speed (m/s)")
    winddir: float = Field(description="Wind direction (degrees)")
    temp: float = Field(description="Air temperature (°C)")
    rh: float = Field(description="Relative humidity (%RH)")
    pressure: float = Field(description="Atmospheric pressure (hPa)")
    rain: float = Field(description="Rainfall (mm)")
    solrad: float = Field(description="Solar radiation")
    netrad: float = Field(description="Net radiation")
    watertemp: float = Field(description="Water temperature (°C)")
    waterlevel: float = Field(description="Water level (m)")
    ta_min: float = Field(description="Minimum temperature")
    ta_max: float = Field(description="Maximum temperature")
    pancilevel: float = Field(description="Pan level")
    pancitemp: float = Field(description="Pan temperature")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WeatherDataResponse(BaseModel):
    data: List[WeatherData]
    count: int
    latest: Optional[WeatherData]

class WeatherSummary(BaseModel):
    parameter: str
    current_value: float
    unit: str
    min_24h: float
    max_24h: float
    avg_24h: float
    status: str

class DataCollectionStatus(BaseModel):
    last_collection: Optional[datetime]
    total_records: int
    collection_active: bool
    collection_interval: int = 60  # seconds
    last_error: Optional[str]

class DateRangeQuery(BaseModel):
    start_date: datetime
    end_date: datetime
    parameters: Optional[List[str]] = None

# Global variables for data collection
collection_status = DataCollectionStatus(
    last_collection=None,
    total_records=0,
    collection_active=False,
    last_error=None
)

# Data collection service
async def fetch_weather_data():
    """Fetch data from weather API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(WEATHER_API_URL) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    raise Exception(f"API returned status {response.status}")
    except Exception as e:
        logging.error(f"Error fetching weather data: {str(e)}")
        return None

async def store_weather_data(weather_json: dict):
    """Store weather data in MongoDB"""
    try:
        # Parse the timestamp
        waktu_str = weather_json.get('waktu')
        if waktu_str:
            # Convert to datetime object
            waktu_dt = datetime.strptime(waktu_str, "%Y-%m-%d %H:%M:%S")
            waktu_dt = waktu_dt.replace(tzinfo=timezone.utc)
        else:
            waktu_dt = datetime.now(timezone.utc)
        
        # Create weather data object
        weather_data = WeatherData(
            idaws=weather_json.get('idaws', ''),
            waktu=waktu_dt,
            windspeed=float(weather_json.get('windspeed', 0)),
            winddir=float(weather_json.get('winddir', 0)),
            temp=float(weather_json.get('temp', 0)),
            rh=float(weather_json.get('rh', 0)),
            pressure=float(weather_json.get('pressure', 0)),
            rain=float(weather_json.get('rain', 0)),
            solrad=float(weather_json.get('solrad', 0)),
            netrad=float(weather_json.get('netrad', 0)),
            watertemp=float(weather_json.get('watertemp', 0)),
            waterlevel=float(weather_json.get('waterlevel', 0)),
            ta_min=float(weather_json.get('ta_min', 0)),
            ta_max=float(weather_json.get('ta_max', 0)),
            pancilevel=float(weather_json.get('pancilevel', 0)),
            pancitemp=float(weather_json.get('pancitemp', 0))
        )
        
        # Check if this timestamp already exists to avoid duplicates
        existing = await db.weather_data.find_one({
            "waktu": waktu_dt,
            "idaws": weather_data.idaws
        })
        
        if not existing:
            # Store in database
            result = await db.weather_data.insert_one(weather_data.dict())
            if result.inserted_id:
                logging.info(f"Stored weather data for {waktu_dt}")
                return True
        else:
            logging.info(f"Data for {waktu_dt} already exists, skipping")
            
        return False
        
    except Exception as e:
        logging.error(f"Error storing weather data: {str(e)}")
        return False

async def collect_weather_data():
    """Main data collection loop"""
    global collection_status
    
    while collection_status.collection_active:
        try:
            # Fetch data from API
            weather_json = await fetch_weather_data()
            
            if weather_json:
                # Store in database
                stored = await store_weather_data(weather_json)
                
                # Update collection status
                collection_status.last_collection = datetime.now(timezone.utc)
                if stored:
                    collection_status.total_records += 1
                collection_status.last_error = None
                
            else:
                collection_status.last_error = "Failed to fetch weather data from API"
                
        except Exception as e:
            collection_status.last_error = str(e)
            logging.error(f"Error in data collection loop: {str(e)}")
        
        # Wait for next collection interval
        await asyncio.sleep(collection_status.collection_interval)

# Routes
@api_router.get("/")
async def root():
    return {
        "message": "Weather Monitoring & Prediction System API",
        "station": "AWS Maritim Pontianak",
        "version": "1.0.0"
    }

@api_router.post("/data-collection/start")
async def start_data_collection(background_tasks: BackgroundTasks):
    """Start automatic data collection"""
    global collection_status
    
    if not collection_status.collection_active:
        collection_status.collection_active = True
        background_tasks.add_task(collect_weather_data)
        
        return {
            "message": "Data collection started",
            "interval": collection_status.collection_interval,
            "status": "active"
        }
    else:
        return {
            "message": "Data collection is already active",
            "status": "active"
        }

@api_router.post("/data-collection/stop")
async def stop_data_collection():
    """Stop automatic data collection"""
    global collection_status
    
    collection_status.collection_active = False
    
    return {
        "message": "Data collection stopped",
        "status": "inactive"
    }

@api_router.get("/data-collection/status")
async def get_collection_status():
    """Get current data collection status"""
    # Update total records count
    total_count = await db.weather_data.count_documents({})
    collection_status.total_records = total_count
    
    return collection_status

@api_router.get("/weather/latest")
async def get_latest_weather():
    """Get the most recent weather data"""
    try:
        latest_data = await db.weather_data.find_one(
            sort=[("waktu", -1)]
        )
        
        if latest_data:
            return WeatherData(**latest_data)
        else:
            raise HTTPException(status_code=404, detail="No weather data found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/weather/historical")
async def get_historical_weather(query: DateRangeQuery):
    """Get historical weather data within date range"""
    try:
        # Build query filter
        filter_query = {
            "waktu": {
                "$gte": query.start_date,
                "$lte": query.end_date
            }
        }
        
        # Get data from database
        cursor = db.weather_data.find(filter_query).sort("waktu", 1)
        data_list = await cursor.to_list(length=None)
        
        # Convert to WeatherData objects
        weather_data = [WeatherData(**item) for item in data_list]
        
        # Get latest data for comparison
        latest_data = None
        if weather_data:
            latest_data = weather_data[-1]
        
        return WeatherDataResponse(
            data=weather_data,
            count=len(weather_data),
            latest=latest_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/weather/summary")
async def get_weather_summary():
    """Get current weather summary with 24-hour statistics"""
    try:
        # Get latest data
        latest_data = await db.weather_data.find_one(
            sort=[("waktu", -1)]
        )
        
        if not latest_data:
            raise HTTPException(status_code=404, detail="No weather data found")
        
        # Get 24-hour data for statistics
        twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
        
        cursor = db.weather_data.find({
            "waktu": {"$gte": twenty_four_hours_ago}
        }).sort("waktu", 1)
        
        historical_data = await cursor.to_list(length=None)
        
        if not historical_data:
            historical_data = [latest_data]  # Use latest if no 24h data
        
        # Calculate statistics for each parameter
        parameters_info = {
            "temp": {"name": "Air Temperature", "unit": "°C", "status_thresholds": {"hot": 35, "warm": 30, "normal": 25, "cool": 20}},
            "rh": {"name": "Relative Humidity", "unit": "%RH", "status_thresholds": {"high": 80, "normal": 60, "low": 40}},
            "pressure": {"name": "Atmospheric Pressure", "unit": "hPa", "status_thresholds": {"high": 1020, "normal": 1013, "low": 1000}},
            "windspeed": {"name": "Wind Speed", "unit": "m/s", "status_thresholds": {"strong": 10, "moderate": 5, "light": 2}},
            "winddir": {"name": "Wind Direction", "unit": "°", "status_thresholds": {}},
            "rain": {"name": "Rainfall", "unit": "mm", "status_thresholds": {"heavy": 10, "moderate": 2, "light": 0.5}},
            "waterlevel": {"name": "Water Level", "unit": "m", "status_thresholds": {"high": 3, "normal": 2, "low": 1}},
            "watertemp": {"name": "Water Temperature", "unit": "°C", "status_thresholds": {"warm": 30, "normal": 25, "cool": 20}},
            "solrad": {"name": "Solar Radiation", "unit": "W/m²", "status_thresholds": {"high": 800, "moderate": 400, "low": 100}},
            "netrad": {"name": "Net Radiation", "unit": "W/m²", "status_thresholds": {}}
        }
        
        summary = []
        
        for param_key, param_info in parameters_info.items():
            if param_key in latest_data:
                # Get values for this parameter
                values = [item.get(param_key, 0) for item in historical_data if item.get(param_key) is not None]
                
                if values:
                    current_value = latest_data[param_key]
                    min_val = min(values)
                    max_val = max(values)
                    avg_val = sum(values) / len(values)
                    
                    # Determine status
                    status = "normal"
                    thresholds = param_info["status_thresholds"]
                    
                    if param_key == "temp":
                        if current_value >= thresholds["hot"]:
                            status = "hot"
                        elif current_value >= thresholds["warm"]:
                            status = "warm"
                        elif current_value <= thresholds["cool"]:
                            status = "cool"
                    elif param_key == "rh":
                        if current_value >= thresholds["high"]:
                            status = "high"
                        elif current_value <= thresholds["low"]:
                            status = "low"
                    elif param_key == "windspeed":
                        if current_value >= thresholds["strong"]:
                            status = "strong"
                        elif current_value >= thresholds["moderate"]:
                            status = "moderate"
                        elif current_value <= thresholds["light"]:
                            status = "light"
                    elif param_key == "rain":
                        if current_value >= thresholds["heavy"]:
                            status = "heavy"
                        elif current_value >= thresholds["moderate"]:
                            status = "moderate"
                        elif current_value >= thresholds["light"]:
                            status = "light"
                        else:
                            status = "no rain"
                    
                    summary.append(WeatherSummary(
                        parameter=param_info["name"],
                        current_value=current_value,
                        unit=param_info["unit"],
                        min_24h=min_val,
                        max_24h=max_val,
                        avg_24h=avg_val,
                        status=status
                    ))
        
        return {
            "station_id": latest_data.get("idaws", ""),
            "station_name": "AWS Maritim Pontianak",
            "last_updated": latest_data.get("waktu"),
            "summary": summary,
            "data_points_24h": len(historical_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/weather/parameters/{parameter}")
async def get_parameter_data(parameter: str, hours: int = 24):
    """Get time series data for a specific parameter"""
    try:
        # Validate parameter
        valid_parameters = ["temp", "rh", "pressure", "windspeed", "winddir", "rain", "waterlevel", "watertemp", "solrad", "netrad"]
        if parameter not in valid_parameters:
            raise HTTPException(status_code=400, detail=f"Invalid parameter. Must be one of: {valid_parameters}")
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # Get data from database
        cursor = db.weather_data.find({
            "waktu": {"$gte": start_time, "$lte": end_time}
        }).sort("waktu", 1)
        
        data_list = await cursor.to_list(length=None)
        
        # Extract parameter values
        time_series = []
        for item in data_list:
            if parameter in item and item[parameter] is not None:
                time_series.append({
                    "timestamp": item["waktu"],
                    "value": item[parameter]
                })
        
        return {
            "parameter": parameter,
            "time_range_hours": hours,
            "data_points": len(time_series),
            "data": time_series
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

@app.on_event("startup")
async def startup_event():
    logger.info("Weather Monitoring System starting up...")
    
    # Create indexes for better query performance
    await db.weather_data.create_index("waktu")
    await db.weather_data.create_index([("idaws", 1), ("waktu", -1)])
    
    logger.info("Database indexes created")