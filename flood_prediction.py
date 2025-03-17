"""
Flood Prediction Module

This module provides functions for predicting water levels based on river data,
weather conditions, and historical measurements.
"""

import os
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import json
import calendar

def get_rivers_and_stations(data_file='combined_river_data.csv'):
    """
    Load river and station data from CSV file.
    
    Args:
        data_file: Path to the CSV file containing river and station data
        
    Returns:
        Dictionary mapping river names to lists of station names
    """
    try:
        data = pd.read_csv(data_file)
        river_stations = data.groupby('River Name')['Station Name'].unique().to_dict()
        
        # Convert numpy arrays to lists for JSON serialization
        for river, stations in river_stations.items():
            river_stations[river] = stations.tolist()
            
        return river_stations
    except Exception as e:
        print(f"Error loading river data: {e}")
        return {}

def get_station_coordinates():
    """
    Get river station coordinates.
    
    Returns:
        Nested dictionary with river names as keys, containing dictionaries 
        with station names as keys and [lat, lon] coordinates as values.
    """
    # This would typically come from your dataset or a separate file
    # Format: {river_name: {station_name: [latitude, longitude]}}
    # For demonstration, using approximate coordinates for Sri Lankan rivers
    return {
        "Kelani": {
            "Nagalagam Street": [6.9271, 79.8612],
            "Hanwella": [6.9022, 80.0850],
            "Glencourse": [6.9784, 80.1417],
            "Kithulgala": [6.9900, 80.4100],
            "Holombuwa": [7.0456, 80.2958],
            "Deraniyagala": [6.9333, 80.3333],
            "Norwood": [6.8350, 80.6150]
        },
        "Kalu": {
            "Putupaula Station": [6.7167, 80.3833],
            "Ellagawa Station": [6.6500, 80.3167],
            "Rathnapura Station": [6.6803, 80.4028],
            "Magura Station": [6.6500, 80.2500],
            "Millakanda Station": [6.6333, 80.1167]
        },
        "Mahaweli": {
            "Manampitiya Station": [7.9167, 81.1000],
            "Weraganthota Station": [7.7667, 80.9500],
            "Peradeniya Station": [7.2667, 80.6000],
            "Nawalapitiya Station": [7.0500, 80.5333],
            "Thaldena Station": [7.1333, 80.7000]
        },
        "Gin": {
            "Baddegama Station": [6.1833, 80.2000],
            "Thawalama Station": [6.3333, 80.3333]
        },
        "Nilwala": {
            "Thalgahagoda Station": [6.0333, 80.5000],
            "Panadugama Station": [6.0500, 80.4833],
            "Pitabeddara Station": [6.1000, 80.4667],
            "Urawa Station": [6.0667, 80.5167]
        },
        "Walawe": {
            "Moraketiya Station": [6.2833, 80.8333]
        }
    }

# OpenWeather API key - replace with your actual API key
OPENWEATHER_API_KEY = "d1afd7c2b937fc33e4c97fe5f0fa25af"

def get_weather_data(lat, lon):
    """
    Get current weather and 5-day forecast from OpenWeather API.
    
    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate
        
    Returns:
        Dictionary containing current weather conditions and forecast data
    """
    try:
        # Current weather
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        current_response = requests.get(current_url)
        current_data = current_response.json()
        
        # 5-day forecast
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()
        
        # Extract relevant data
        if current_response.status_code == 200 and forecast_response.status_code == 200:
            # Current rainfall (convert from mm/3h to mm/day if needed)
            current_rainfall = current_data.get('rain', {}).get('1h', 0) * 24  # Approximate daily rainfall
            if current_rainfall == 0:
                current_rainfall = current_data.get('rain', {}).get('3h', 0) * 8  # Alternative calculation
            
            # Previous 5 days rainfall (approximated from forecast data)
            # In a real app, you would use historical data API
            prev_rainfall = [0] * 5  # Placeholder for previous rainfall
            
            return {
                'current': {
                    'temp': current_data['main']['temp'],
                    'humidity': current_data['main']['humidity'],
                    'rainfall': current_rainfall,
                    'description': current_data['weather'][0]['description'],
                    'icon': current_data['weather'][0]['icon']
                },
                'forecast': forecast_data['list'][:5],  # Next 5 forecast points
                'prev_rainfall': prev_rainfall
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def predict_water_level(river_name, station_name, month, day, rainfall, prev_rainfall, prev_water_levels, model_path='water_level_prediction_model.pkl'):
    """
    Predict water level based on input parameters.
    
    Args:
        river_name: Name of the river
        station_name: Name of the station
        month: Month name (e.g., 'January')
        day: Day of month (1-31)
        rainfall: Current rainfall in mm
        prev_rainfall: List of previous 5 days' rainfall values
        prev_water_levels: List of previous 5 days' water level values
        model_path: Path to the prediction model file
        
    Returns:
        Dictionary with prediction result and risk level
    """
    # Check if model exists
    if not os.path.exists(model_path):
        return {"error": "Model file not found. Please train the model first."}
    
    try:
        # Load the model pipeline
        pipeline = joblib.load(model_path)
        
        # Validate and convert month to month number
        try:
            month_num = pd.to_datetime(month, format='%B').month
        except:
            print(f"Invalid month: {month}, using January")
            month = 'January'
            month_num = 1
        
        # Validate day is within range for the month
        max_days = {
            'January': 31, 'February': 29, 'March': 31, 'April': 30,
            'May': 31, 'June': 30, 'July': 31, 'August': 31,
            'September': 30, 'October': 31, 'November': 30, 'December': 31
        }
        
        if day > max_days.get(month, 31):
            print(f"Day {day} exceeds maximum for {month}, using {max_days.get(month, 31)}")
            day = max_days.get(month, 31)
        
        # Calculate day of year
        try:
            day_of_year = pd.to_datetime(f"2023-{month_num}-{day}").dayofyear
        except:
            day_of_year = 1
        
        # Ensure we have enough lag values
        if len(prev_rainfall) < 5:
            prev_rainfall = prev_rainfall + [0] * (5 - len(prev_rainfall))
        if len(prev_water_levels) < 5:
            prev_water_levels = prev_water_levels + [3.0] * (5 - len(prev_water_levels))
        
        # Limit to 5 values
        prev_rainfall = prev_rainfall[:5]
        prev_water_levels = prev_water_levels[:5]
        
        # Create a dataframe with the input data
        data = {
            'River Name': [river_name],
            'Station Name': [station_name],
            'Month': [month],
            'Month_num': [month_num],
            'Day': [day],
            'Day_of_year': [day_of_year],
            'Year (2023) - Rainfall (mm)': [rainfall]
        }
        
        # Add lag features
        for i, (rain, level) in enumerate(zip(prev_rainfall, prev_water_levels), 1):
            data[f'Year (2023) - Rainfall (mm)_lag_{i}'] = [rain]
            data[f'Year (2023) - Water Level (m)_lag_{i}'] = [level]
        
        # Create rolling features
        data[f'Year (2023) - Rainfall (mm)_roll_mean_3'] = [np.mean(prev_rainfall[:3] + [rainfall])]
        data[f'Year (2023) - Rainfall (mm)_roll_std_3'] = [np.std(prev_rainfall[:3] + [rainfall])]
        data[f'Year (2023) - Water Level (m)_roll_mean_3'] = [np.mean(prev_water_levels[:3])]
        data[f'Year (2023) - Water Level (m)_roll_std_3'] = [np.std(prev_water_levels[:3])]
        
        data[f'Year (2023) - Rainfall (mm)_roll_mean_7'] = [np.mean(prev_rainfall + [rainfall])]
        data[f'Year (2023) - Rainfall (mm)_roll_std_7'] = [np.std(prev_rainfall + [rainfall])]
        data[f'Year (2023) - Water Level (m)_roll_mean_7'] = [np.mean(prev_water_levels)]
        data[f'Year (2023) - Water Level (m)_roll_std_7'] = [np.std(prev_water_levels)]
        
        # Create a dataframe
        df = pd.DataFrame(data)
        
        # Create river-specific base levels for better variation
        river_base_levels = {
            "Kelani": 3.2,
            "Kalu": 2.8,
            "Mahaweli": 3.5,
            "Gin": 2.5,
            "Nilwala": 2.7,
            "Walawe": 3.0
        }
        
        # Get base level for current river or default to 3.0
        river_base = river_base_levels.get(river_name, 3.0)
        
        # Make prediction using the pipeline
        base_prediction = float(pipeline.predict(df)[0])
        
        # First, check if the base prediction already indicates high risk - if so, return it directly
        if base_prediction >= 5.0:
            # For high risk, return consistent prediction without randomization
            # Safety critical data should be consistent
            return {
                "prediction": round(base_prediction, 2),
                "risk_level": "HIGH",
                "river_base_level": river_base
            }
        
        # Calculate seasonal factor (higher in rainy seasons)
        rainy_months = [5, 6, 10, 11, 12]  # May, June, October, November, December
        seasonal_factor = 0.3 if month_num in rainy_months else 0.1
        
        # Add variation based on input parameters for non-high risk predictions
        avg_rainfall = np.mean(prev_rainfall + [rainfall])
        avg_water_level = np.mean(prev_water_levels)
        rainfall_trend = rainfall - (np.mean(prev_rainfall) if prev_rainfall else 0)
        
        # Different variation levels based on risk levels
        # For lower risk levels, add more randomization
        if base_prediction < 3.5:  # Normal risk
            # More randomization for normal risk
            variation_factor = 0.3
        elif base_prediction < 4.0:  # Moderate-low risk
            # Moderate randomization
            variation_factor = 0.2
        else:  # Moderate risk
            # Less randomization as risk increases
            variation_factor = 0.1
            
        # Calculate a variation factor based on risk level
        variation = (np.random.random() - 0.5) * variation_factor
        
        # Apply rain factor - higher rainfall means more possible variation
        if avg_rainfall > 20:
            rain_factor = min(0.5, avg_rainfall / 80)  # Cap at 0.5
            rainfall_variation = (np.random.random() - 0.5) * rain_factor
        else:
            rainfall_variation = 0
        
        # Apply modifiers
        modifiers = [
            river_base * 0.1,  # River base contribution
            rainfall_trend * 0.05,  # Rainfall trend contribution
            seasonal_factor,  # Seasonal factor
            variation,  # Risk-based variation
            rainfall_variation  # Rainfall-based variation
        ]
        
        # Calculate final prediction with modifiers
        prediction = base_prediction + sum(modifiers)
        
        # Constrain prediction to reasonable bounds (2.0 to 6.0)
        prediction = max(2.0, min(6.0, prediction))
        
        # Determine risk level based on final prediction
        risk_level = "NORMAL"
        if prediction > 5.0:
            risk_level = "HIGH"
            # Stabilize predictions for HIGH risk - safety critical
            prediction = max(5.0, base_prediction)  # Ensure minimum of 5.0
        elif prediction > 4.0:
            risk_level = "MODERATE"
        elif prediction > 3.5:
            risk_level = "MODERATE-LOW"
        elif prediction > 3.0:
            risk_level = "LOW"
        
        return {
            "prediction": round(prediction, 2),
            "risk_level": risk_level,
            "river_base_level": river_base
        }
    except Exception as e:
        print(f"Prediction error details: {str(e)}")
        # Print the traceback for more detailed error information
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction error: {str(e)}"}

def reset_flood_model(model_path='water_level_prediction_model.pkl'):
    """
    Reset the water level prediction model by creating a new default model.
    
    Args:
        model_path: Path where the model will be saved
        
    Returns:
        Dictionary indicating success or failure
    """
    try:
        # Delete the existing model
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Create a new model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # Generate sample data
        n_samples = 100
        
        # Get categorical values
        rivers = list(get_station_coordinates().keys())
        stations = [station for river in get_station_coordinates().values() for station in river.keys()]
        months = list(calendar.month_name)[1:]
        
        # Create a DataFrame with all features
        X_dummy = pd.DataFrame({
            'River Name': np.random.choice(rivers, size=n_samples),
            'Station Name': np.random.choice(stations, size=n_samples),
            'Month': np.random.choice(months, size=n_samples),
            'Month_num': np.random.randint(1, 13, size=n_samples),
            'Day': np.random.randint(1, 32, size=n_samples),
            'Day_of_year': np.random.randint(1, 367, size=n_samples),
            'Year (2023) - Rainfall (mm)': np.random.rand(n_samples) * 50,  # 0-50mm rainfall
        })
        
        # Add lag features
        for i in range(1, 6):
            X_dummy[f'Year (2023) - Rainfall (mm)_lag_{i}'] = np.random.rand(n_samples) * 50
            X_dummy[f'Year (2023) - Water Level (m)_lag_{i}'] = 2.0 + np.random.rand(n_samples) * 4.0
        
        # Add rolling features
        for window in [3, 7]:
            X_dummy[f'Year (2023) - Rainfall (mm)_roll_mean_{window}'] = np.random.rand(n_samples) * 50
            X_dummy[f'Year (2023) - Rainfall (mm)_roll_std_{window}'] = np.random.rand(n_samples) * 10
            X_dummy[f'Year (2023) - Water Level (m)_roll_mean_{window}'] = 2.0 + np.random.rand(n_samples) * 4.0
            X_dummy[f'Year (2023) - Water Level (m)_roll_std_{window}'] = np.random.rand(n_samples)
        
        # Define categorical and numerical columns
        categorical_cols = ['River Name', 'Station Name', 'Month']
        numerical_cols = [col for col in X_dummy.columns if col not in categorical_cols]
        
        # Create preprocessing steps
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=10, random_state=42))
        ])
        
        # Generate target values (water levels between 2.0 and 6.0)
        y_dummy = 2.0 + 4.0 * np.random.rand(n_samples)
        
        # Train the model
        pipeline.fit(X_dummy, y_dummy)
        
        # Save the model
        joblib.dump(pipeline, model_path)
        
        print(f"Created a new water level prediction model at {model_path}")
        
        return {
            "success": True,
            "message": "Water level prediction model has been reset successfully."
        }
    except Exception as e:
        print(f"Error resetting flood model: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to reset model: {str(e)}"
        }

# Function to calculate the Haversine distance between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth.
    
    Args:
        lat1, lon1: Coordinates of first point in decimal degrees
        lat2, lon2: Coordinates of second point in decimal degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r 