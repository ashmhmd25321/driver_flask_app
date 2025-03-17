import os
from flask import Flask, Response, request, jsonify, render_template, session
import numpy as np
import pandas as pd
import joblib
import requests
from keras.models import load_model
from keras_preprocessing.image import load_img, img_to_array
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import soundfile as sf
import cv2
import time
import uuid
from datetime import datetime, timedelta
import json
from threading import Lock
import warnings
from flask_cors import CORS
import subprocess
import sys
import calendar
# Import the flood_prediction module
import flood_prediction

# Suppress warnings
warnings.filterwarnings('ignore')

# Get the absolute path to the application directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define directories
TEMP_DIR = os.path.join(APP_ROOT, 'temp')
STATIC_DIR = os.path.join(APP_ROOT, 'static')

# Create directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Check and install required packages
def check_and_install_requirements():
    required_packages = [
        'flask', 'numpy', 'pandas', 'joblib', 'requests', 'tensorflow', 'keras',
        'scikit-learn', 'librosa', 'soundfile', 'opencv-python', 'flask-cors', 'gdown'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Successfully installed {package}")
            except Exception as e:
                print(f"✗ Failed to install {package}: {str(e)}")

# Check and install requirements
print("Checking required packages...")
check_and_install_requirements()

# Define paths for models
DRIVER_BEHAVIOR_MODEL_PATH = os.path.join(APP_ROOT, 'vggModel_driver_behaviour.h5')
ROAD_SIGN_MODEL_PATH = os.path.join(APP_ROOT, 'vggModel.h5')
AUDIO_MODEL_PATH = os.path.join(APP_ROOT, 'audio_classification_model.h5')
LABEL_ENCODER_PATH = os.path.join(APP_ROOT, 'label_encoder.pkl')
SCALER_PATH = os.path.join(APP_ROOT, 'scaler.pkl')
WATER_LEVEL_MODEL_PATH = os.path.join(APP_ROOT, 'water_level_prediction_model.pkl')

# Function to download model from Google Drive
def download_from_gdrive(file_id, destination):
    """Download a file from Google Drive using gdown"""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return True
    
    try:
        # Check if gdown is installed
        try:
            import gdown
        except ImportError:
            print("Installing gdown...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        print(f"Downloading file from Google Drive to {destination}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        
        if os.path.exists(destination):
            print(f"Successfully downloaded: {destination}")
            return True
        else:
            print(f"Failed to download: {destination}")
            return False
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return False

# Download models if they don't exist
if not os.path.exists(DRIVER_BEHAVIOR_MODEL_PATH):
    print("Driver behavior model not found. Downloading from Google Drive...")
    download_from_gdrive("1ZIyCdv4bkyOJ47S9lLiZqD6mnqGBFcIe", DRIVER_BEHAVIOR_MODEL_PATH)

if not os.path.exists(ROAD_SIGN_MODEL_PATH):
    print("Road sign model not found. Downloading from Google Drive...")
    download_from_gdrive("1Cs5QlqANw9cjI9plj4XLtgvdpXu0Ww90", ROAD_SIGN_MODEL_PATH)

if not os.path.exists(AUDIO_MODEL_PATH):
    print("Audio model not found. Downloading from Google Drive...")
    download_from_gdrive("1cLDCrlTctFcZ74VB0Sa02sYs6HJBgw7Z", AUDIO_MODEL_PATH)

# Download label encoder and scaler if they don't exist
# Note: You'll need to provide the correct file IDs for these files
if not os.path.exists(LABEL_ENCODER_PATH):
    print("Label encoder not found. Creating a default one...")
    # Create a default label encoder with common audio classes
    from sklearn.preprocessing import LabelEncoder
    default_classes = ["car_horn", "engine_idling", "gun_shot", "siren", "street_music", "air_conditioner", "children_playing", "dog_bark", "drilling", "jackhammer"]
    le = LabelEncoder()
    le.fit(default_classes)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)

if not os.path.exists(SCALER_PATH):
    print("Scaler not found. Creating a default one...")
    # Create a default scaler
    default_scaler = StandardScaler()
    # Initialize with some reasonable values for audio features
    default_features = np.random.rand(10, 193)  # Typical MFCC feature size
    default_scaler.fit(default_features)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(default_scaler, f)

# Create a default water level prediction model if it doesn't exist
if not os.path.exists(WATER_LEVEL_MODEL_PATH):
    print("Water level prediction model not found. Creating a default one...")
    flood_prediction.reset_flood_model(WATER_LEVEL_MODEL_PATH)

# Load the trained models
print("Loading models...")
try:
    model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    print("Audio model loaded successfully")
except Exception as e:
    print(f"Error loading audio model: {str(e)}")
    model = None

try:
    model1 = load_model(DRIVER_BEHAVIOR_MODEL_PATH, compile=False)
    print("Driver behavior model loaded successfully")
except Exception as e:
    print(f"Error loading driver behavior model: {str(e)}")
    model1 = None

try:
    model2 = load_model(ROAD_SIGN_MODEL_PATH, compile=False)
    print("Road sign model loaded successfully")
except Exception as e:
    print(f"Error loading road sign model: {str(e)}")
    model2 = None

class_names = [
    "safe driving",
    "texting - right",
    "talking on the phone - right",
    "texting - left",
    "talking on the phone - left",
    "operating the radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger"
]

class_names_sign = [
    "children crossing",
    "hospital",
    "level crossing with gates",
    "no honking",
    "no left turn",
    "no right turn",
    "no u turn"
]

# Load the label encoder and scaler
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Label encoder loaded successfully")
except Exception as e:
    print(f"Error loading label encoder: {str(e)}")
    label_encoder = None

try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {str(e)}")
    scaler = None

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)
app.secret_key = os.urandom(24)


# Load river and station data
def get_rivers_and_stations():
    try:
        data = pd.read_csv(os.path.join(APP_ROOT, 'combined_river_data.csv'))
        river_stations = data.groupby('River Name')['Station Name'].unique().to_dict()
        
        # Convert numpy arrays to lists for JSON serialization
        for river, stations in river_stations.items():
            river_stations[river] = stations.tolist()
            
        return river_stations
    except Exception as e:
        print(f"Error loading river data: {e}")
        return {}
    
# Get river station coordinates (for demonstration, you would replace with actual coordinates)
def get_station_coordinates():
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
        # Add coordinates for other rivers and stations
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
    """Get current weather and 5-day forecast from OpenWeather API"""
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

def predict_water_level(river_name, station_name, month, day, rainfall, prev_rainfall, prev_water_levels):
    """
    Predict water level based on input parameters.
    """
    # Check if model exists
    model_path = os.path.join(APP_ROOT, 'water_level_prediction_model.pkl')
    if not os.path.exists(model_path):
        return {"error": "Model file not found. Please train the model first."}
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Validate and convert month to month number
        try:
            month_num = pd.to_datetime(month, format='%B').month
        except:
            month = 'January'
            month_num = 1
        
        # Validate day is within range for the month
        max_days = {
            'January': 31, 'February': 29, 'March': 31, 'April': 30,
            'May': 31, 'June': 30, 'July': 31, 'August': 31,
            'September': 30, 'October': 31, 'November': 30, 'December': 31
        }
        
        if day > max_days.get(month, 31):
            day = max_days.get(month, 31)
        
        # Calculate day of year
        try:
            day_of_year = pd.to_datetime(f"2023-{month_num}-{day}").dayofyear
        except:
            day_of_year = 1
        
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
            if i <= 5:  # We only use 5 lag features
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
        
        # Make prediction
        prediction = float(model.predict(df)[0])
        
        # Determine risk level
        risk_level = "NORMAL"
        if prediction > 5.0:
            risk_level = "HIGH"
        elif prediction > 4.0:
            risk_level = "MODERATE"
        
        return {
            "prediction": round(prediction, 2),
            "risk_level": risk_level
        }
    except Exception as e:
        print(f"Prediction error details: {str(e)}")
        return {"error": f"Prediction error: {str(e)}"}
    

# Function to calculate the Haversine distance between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
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

@app.route('/floodP')
def indexFP():
    rivers_and_stations = flood_prediction.get_rivers_and_stations(os.path.join(APP_ROOT, 'combined_river_data.csv'))
    station_coordinates = flood_prediction.get_station_coordinates()
    
    # Get current date
    current_date = datetime.now()
    month = current_date.strftime('%B')
    day = current_date.day
    
    return render_template('indexFP.html', 
                          rivers_and_stations=json.dumps(rivers_and_stations),
                          station_coordinates=json.dumps(station_coordinates),
                          current_month=month,
                          current_day=day)


@app.route('/api/stations')
def get_stations():
    rivers_and_stations = flood_prediction.get_rivers_and_stations(os.path.join(APP_ROOT, 'combined_river_data.csv'))
    return jsonify(rivers_and_stations)

@app.route('/api/weather')
def get_weather():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if not lat or not lon:
        return jsonify({"error": "Latitude and longitude are required"}), 400
    
    weather_data = flood_prediction.get_weather_data(lat, lon)
    if weather_data:
        return jsonify(weather_data)
    else:
        return jsonify({"error": "Failed to fetch weather data"}), 500

@app.route('/api/predict', methods=['POST'])
def predictFP():
    try:
        data = request.json
        
        river_name = data.get('river_name')
        station_name = data.get('station_name')
        month = data.get('month')
        day = data.get('day', 1)
        rainfall = data.get('rainfall', 0)
        prev_rainfall = data.get('prev_rainfall', [0, 0, 0, 0, 0])
        prev_water_levels = data.get('prev_water_levels', [3.0, 3.0, 3.0, 3.0, 3.0])
        
        # Validate inputs
        if not all([river_name, station_name, month]):
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Make prediction
        result = flood_prediction.predict_water_level(
            river_name=river_name,
            station_name=station_name,
            month=month,
            day=int(day),
            rainfall=float(rainfall),
            prev_rainfall=[float(r) for r in prev_rainfall],
            prev_water_levels=[float(w) for w in prev_water_levels],
            model_path=WATER_LEVEL_MODEL_PATH
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/nearest-station')
def find_nearest_station():
    """Find the nearest river station to the given coordinates"""
    try:
        # Get latitude and longitude from request
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        if not lat or not lng:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        
        # Get station coordinates
        stations_data = flood_prediction.get_station_coordinates()
        
        # Find the nearest station
        nearest_river = None
        nearest_station = None
        nearest_coords = None
        min_distance = float('inf')
        
        for river, stations in stations_data.items():
            for station, coords in stations.items():
                station_lat, station_lng = coords
                distance = flood_prediction.haversine_distance(lat, lng, station_lat, station_lng)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_river = river
                    nearest_station = station
                    nearest_coords = coords
        
        if nearest_river and nearest_station:
            return jsonify({
                "river": nearest_river,
                "station": nearest_station,
                "station_lat": nearest_coords[0],
                "station_lng": nearest_coords[1],
                "distance_km": round(min_distance, 2)
            })
        else:
            return jsonify({"error": "No river stations found"}), 404
            
    except Exception as e:
        print(f"Error finding nearest station: {e}")
        return jsonify({"error": str(e)}), 500


# Global variable to store video sources
video_sources = {}
# Lock for thread safety
video_sources_lock = Lock()

def extract_features(audio_path, n_mfcc=13):
    try:
        # Try loading with soundfile first (better for various formats)
        try:
            y, sr = sf.read(audio_path)
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
        except Exception:
            # Fall back to librosa if soundfile fails
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Resample if needed
        if sr != 44100:
            y = librosa.resample(y, orig_sr=sr, target_sr=44100)
            sr = 44100
        
        # Ensure audio is not empty
        if len(y) == 0:
            raise ValueError("Audio file is empty or could not be read properly")
            
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Trim silence
        try:
            y, _ = librosa.effects.trim(y, top_db=20)
        except Exception:
            # If trimming fails, continue with original audio
            pass
        
        # Ensure minimum length (1 second)
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)), 'constant')
            
        # Extract features with error handling
        try:
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        except Exception:
            # Use zeros if feature extraction fails
            mfccs = np.zeros(n_mfcc)
            
        try:
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        except Exception:
            chroma = np.zeros(12)  # Default chroma size
            
        try:
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        except Exception:
            spectral_contrast = np.zeros(7)  # Default spectral contrast size
            
        try:
            zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        except Exception:
            zcr = np.zeros(1)
            
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        except Exception:
            spectral_centroid = np.zeros(1)
            
        try:
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
        except Exception:
            spectral_rolloff = np.zeros(1)
        
        # Combine all features
        features = np.hstack([mfccs, chroma, spectral_contrast, zcr, spectral_centroid, spectral_rolloff])
        
        # Check for NaN or infinity values and replace with zeros
        features = np.nan_to_num(features)
        
        return features
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        raise ValueError(f"Error processing audio file: {str(e)}")
    
def predict_signRS(image_path):
    if model2 is None:
        return "Model not loaded"
    
    img = load_img(image_path, target_size=(150, 150, 3))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model2.predict(img)
    predicted_class_index = np.argmax(prediction, axis=-1)[0]
    result = class_names_sign[predicted_class_index]
    return result


# Function to predict from numpy array (for video frames)
def predict_from_array(img_array, model_type='road_sign'):
    if model_type == 'road_sign':
        if model2 is None:
            return "Model not loaded", 0.0
            
        # Resize the image to match model input size
        img = cv2.resize(img_array, (150, 150))
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize the image
        img = img / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model2.predict(img)
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        confidence = float(prediction[0][predicted_class_index])
        result = class_names_sign[predicted_class_index]
    
    elif model_type == 'driver_behavior':
        if model1 is None:
            return "Model not loaded", 0.0
            
        # Resize the image to match model input size
        img = cv2.resize(img_array, (224, 224))
        # Convert BGR to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize the image
        img = img / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = model1.predict(img)
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        confidence = float(prediction[0][predicted_class_index])
        result = class_names[predicted_class_index]
    
    return result, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/roadsign')
def sign():
    return render_template('indexRS.html')

@app.route('/video')
def video():
    # Generate a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Default to webcam
    with video_sources_lock:
        if session['user_id'] not in video_sources:
            video_sources[session['user_id']] = 0
        
    return render_template('video.html')

@app.route('/driverBehaviour')
def index():
    return render_template('indexD.html')

@app.route('/driver_video')
def driver_video():
    # Generate a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Default to webcam
    with video_sources_lock:
        if session['user_id'] not in video_sources:
            video_sources[session['user_id']] = 0
        
    return render_template('driver_video.html')

@app.route('/predictRS', methods=['POST'])
def predictRS():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Check if model is loaded
        if model2 is None:
            return jsonify({"error": "Road sign model not loaded. Please check server logs."}), 500
            
        image_path = os.path.join(TEMP_DIR, "uploaded_image_rs.jpg")
        image.save(image_path)

        # Load and preprocess the image
        img = load_img(image_path, target_size=(150, 150, 3))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model2.predict(img)
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        confidence = float(prediction[0][predicted_class_index])
        result = class_names_sign[predicted_class_index]
        
        # Only return prediction if confidence is above 0.7
        if confidence >= 0.7:
            return jsonify({"prediction": result, "confidence": f"{confidence:.2f}"})
        else:
            return jsonify({"message": "No confident prediction available", "confidence": f"{confidence:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_frames(user_id=None, model_type='road_sign'):
    # Use default webcam if no user_id provided
    video_source = 0
    
    # If user_id is provided, try to get their video source
    if user_id:
        with video_sources_lock:
            video_source = video_sources.get(user_id, 0)
    
    # Open video capture
    cap = cv2.VideoCapture(video_source)
    
    # Check if camera/video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}.")
        # Create static folder if it doesn't exist
        if not os.path.exists(STATIC_DIR):
            os.makedirs(STATIC_DIR)
        
        # Create a default error image if it doesn't exist
        error_img_path = os.path.join(STATIC_DIR, 'error.jpg')
        if not os.path.exists(error_img_path):
            # Create a blank image with error text
            error_img = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(error_img, "Video source not available", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(error_img_path, error_img)
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               open(error_img_path, 'rb').read() + b'\r\n')
        return

    # For limiting prediction frequency
    last_prediction_time = 0
    prediction_interval = 0.5  # seconds between predictions
    current_prediction = "No detection"
    confidence = 0.0


    try:
        while True:
            # Read a frame
            success, frame = cap.read()
            if not success:
                # If video file ended, loop back to beginning for uploaded videos
                if isinstance(video_source, str) and os.path.exists(video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = cap.read()
                    if not success:
                        break
                else:
                    break
            
            # Get current time for prediction throttling
            current_time = time.time()
            
            # Make prediction every prediction_interval seconds
            if current_time - last_prediction_time > prediction_interval:
                try:
                    current_prediction, confidence = predict_from_array(frame, model_type)
                    last_prediction_time = current_time
                except Exception as e:
                    print(f"Prediction error: {e}")
            
            # Add prediction text to frame only if confidence is above 0.7
            if confidence > 0.7:
                cv2.putText(
                    frame, 
                    f"Prediction: {current_prediction} ({confidence:.2f})", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
            else:
                cv2.putText(
                    frame, 
                    "No confident prediction", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        # Release resources
        cap.release()


@app.route('/video_feed')
def video_feed():
    # Get user ID from session within request context
    user_id = session.get('user_id')
    # Pass the user_id to generate_frames
    return Response(generate_frames(user_id, 'road_sign'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/driver_video_feed')
def driver_video_feed():
    # Get user ID from session within request context
    user_id = session.get('user_id')
    # Pass the user_id to generate_frames with driver_behavior model type
    return Response(generate_frames(user_id, 'driver_behavior'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No video selected"}), 400
    
    model_type = request.form.get('model_type', 'road_sign')
    
    try:
        # Generate a unique session ID if not already present
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        # Save the uploaded video with a unique name
        video_filename = f"uploaded_video_{session['user_id']}.mp4"
        video_path = os.path.join(TEMP_DIR, video_filename)
        video.save(video_path)
        
        # Set this user's video source to the uploaded file
        with video_sources_lock:
            video_sources[session['user_id']] = video_path
        
        # Return success response with the path to access the video stream
        video_feed_url = "/driver_video_feed" if model_type == "driver_behavior" else "/video_feed"
        return jsonify({
            "success": True,
            "message": "Video uploaded successfully",
            "video_feed_url": video_feed_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/use_webcam')
def use_webcam():
    # Generate a unique session ID if not already present
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    model_type = request.args.get('model_type', 'road_sign')
    
    # Set this user's video source to webcam
    with video_sources_lock:
        video_sources[session['user_id']] = 0
    
    video_feed_url = "/driver_video_feed" if model_type == "driver_behavior" else "/video_feed"
    return jsonify({
        "success": True,
        "message": "Switched to webcam",
        "video_feed_url": video_feed_url
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    audio_file = request.files['audio_file']
    
    # Check if the file is empty
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Check if model and preprocessing objects are loaded
    if model is None or scaler is None or label_encoder is None:
        return jsonify({'error': 'Audio classification model or preprocessing objects not loaded. Please check server logs.'})
    
    # Save the uploaded file
    audio_path = os.path.join(TEMP_DIR, audio_file.filename)
    audio_file.save(audio_path)

    try:
        # Extract features and make prediction
        features = extract_features(audio_path).reshape(1, -1)
        
        # Check if features match the expected shape
        expected_shape = scaler.n_features_in_
        if features.shape[1] != expected_shape:
            # If shapes don't match, pad or truncate to match expected shape
            if features.shape[1] < expected_shape:
                # Pad with zeros
                padded = np.zeros((1, expected_shape))
                padded[0, :features.shape[1]] = features[0, :]
                features = padded
            else:
                # Truncate
                features = features[:, :expected_shape]
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction)) * 100  # Convert to percentage

        # Only return prediction if confidence is above 70%
        if confidence >= 70:
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': f"{confidence:.2f}%"
            })
        else:
            return jsonify({
                'message': 'No confident prediction available',
                'confidence': f"{confidence:.2f}%"
            })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)})
    finally:
        # Clean up the temporary file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

@app.route('/predictD', methods=['POST'])
def predictD():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Check if model is loaded
        if model1 is None:
            return jsonify({"error": "Driver behavior model not loaded. Please check server logs."}), 500
            
        image_path = os.path.join(TEMP_DIR, "uploaded_image_d.jpg")
        image.save(image_path)

        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224, 3))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model1.predict(img)
        predicted_class_index = np.argmax(prediction, axis=-1)[0]
        confidence = float(prediction[0][predicted_class_index])
        result = class_names[predicted_class_index]
        
        # Only return prediction if confidence is above 0.7
        if confidence >= 0.7:
            return jsonify({"prediction": result, "confidence": f"{confidence:.2f}"})
        else:
            return jsonify({"message": "No confident prediction available", "confidence": f"{confidence:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-permissions')
def test_permissions():
    """Serve a page to test camera, microphone, and location permissions"""
    test_page_path = os.path.join(APP_ROOT, 'test_permissions.html')
    
    # Check if the test page exists
    if os.path.exists(test_page_path):
        with open(test_page_path, 'r') as f:
            content = f.read()
        return content
    else:
        return "Permission test page not found. Please make sure test_permissions.html exists in the application directory."

@app.route('/reset-flood-model')
def reset_flood_model():
    """Reset the water level prediction model"""
    try:
        result = flood_prediction.reset_flood_model(WATER_LEVEL_MODEL_PATH)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to reset model: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Create static folder if it doesn't exist
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
    
    # Create a default error image if it doesn't exist
    error_img_path = os.path.join(STATIC_DIR, 'error.jpg')
    if not os.path.exists(error_img_path):
        # Create a blank image with error text
        error_img = np.zeros((300, 500, 3), dtype=np.uint8)
        cv2.putText(error_img, "Video source not available", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(error_img_path, error_img)
    
    # Check if any models need to be downloaded
    model_files = {
        'vggModel_driver_behaviour.h5': '1ZIyCdv4bkyOJ47S9lLiZqD6mnqGBFcIe',
        'vggModel.h5': '1Cs5QlqANw9cjI9plj4XLtgvdpXu0Ww90',
        'audio_classification_model.h5': '1cLDCrlTctFcZ74VB0Sa02sYs6HJBgw7Z'
    }
    
    for model_file, file_id in model_files.items():
        model_path = os.path.join(APP_ROOT, model_file)
        if not os.path.exists(model_path):
            print(f"Model {model_file} not found. Attempting to download...")
            download_from_gdrive(file_id, model_path)
    
    # For production on EC2
    app.run(host='0.0.0.0', port=5000, debug=False)
