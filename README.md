# Accident ML Models - Flask API

This repository contains a Flask API for various accident prediction and classification models, including:

- Audio Classification
- Driver Behavior Prediction
- Road Sign Prediction
- Flood Prediction

## Features

- Audio classification with real-time recording and prediction
- Image-based driver behavior prediction
- Road sign recognition
- Flood prediction based on weather data
- Modern, responsive UI for all features

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Flask\ API
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

The application will automatically:
- Check and install required packages
- Download necessary models from Google Drive
- Create default models if needed
- Start the server on port 5000

## Accessing the Application

Once running, access the application at:
```
http://localhost:5000
```

For deployment on AWS EC2, access using your instance's public IP:
```
http://<your-ec2-ip>:5000
```

## API Endpoints

- `/` - Audio Classification
- `/driverBehaviour` - Driver Behavior Prediction
- `/roadsign` - Road Sign Prediction
- `/floodP` - Flood Prediction
- `/predict` - Audio Classification API
- `/predictD` - Driver Behavior Prediction API
- `/predictRS` - Road Sign Prediction API

## Models

The application uses the following models:

1. **Audio Classification Model**
   - File: `audio_classification_model.h5`
   - Classifies audio into various categories like car horn, engine idling, etc.

2. **Driver Behavior Model**
   - File: `vggModel_driver_behaviour.h5`
   - Classifies driver behavior into categories like safe driving, texting, etc.

3. **Road Sign Model**
   - File: `vggModel.h5`
   - Recognizes various road signs

4. **Water Level Prediction Model**
   - File: `water_level_prediction_model.pkl`
   - Predicts water levels based on rainfall and other factors

## Deployment on AWS EC2

1. Launch an EC2 instance with Ubuntu
2. Install Python and required dependencies
3. Clone this repository
4. Install requirements: `pip install -r requirements.txt`
5. Run the application: `python app.py`
6. Configure security group to allow inbound traffic on port 5000

For production deployment, consider using Gunicorn:
```
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

[MIT License](LICENSE)
