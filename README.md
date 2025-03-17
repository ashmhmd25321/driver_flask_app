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
- Update HTML files with permission request buttons
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

### Camera, Microphone, and Location Access

The application includes permission request buttons that appear in the top-right corner of each page. These buttons allow you to:

1. **Allow Camera** - Grants access to your device's camera for video features
2. **Allow Microphone** - Grants access to your device's microphone for audio recording
3. **Allow Location** - Grants access to your device's location for geolocation features

Simply click these buttons when you need to use these features. The application will also attempt to request these permissions automatically when appropriate.

### Testing Permissions

To verify that your browser can access your camera, microphone, and location, visit the test page:
```
http://localhost:5000/test-permissions
```

This page provides a simple interface to test each permission individually and see if they're working correctly. Use this page to troubleshoot any permission issues before using the main application.

**Note for EC2 Deployment:** Make sure your EC2 instance's security group allows inbound traffic on port 5000.

## API Endpoints

The application provides the following API endpoints:

- `/api/predict` - POST endpoint for audio classification
- `/api/predict-driver` - POST endpoint for driver behavior prediction
- `/api/predict-water-level` - POST endpoint for water level prediction
- `/api/weather` - GET endpoint for weather data based on coordinates
- `/api/nearest-station` - GET endpoint for finding the nearest river station
- `/test-permissions` - GET endpoint to test camera, microphone, and location permissions
- `/reset-flood-model` - GET endpoint to reset the water level prediction model

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
2. Install Python and required dependencies:
   ```
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-dev
   ```
3. Clone this repository
4. Install requirements: `pip3 install -r requirements.txt`
5. Run the application: `python3 app.py`
6. Configure security group to allow inbound traffic on port 5000

### Troubleshooting Permission Issues

If you encounter issues with camera or microphone access:

1. **Browser Settings**: Make sure your browser settings allow camera and microphone access for the site
2. **Click Permission Buttons**: Use the permission request buttons in the top-right corner of the page
3. **Use Test Page**: Visit `/test-permissions` to verify each permission individually
4. **Clear Browser Cache**: Sometimes clearing your browser cache can resolve permission issues
5. **Try Different Browser**: Some browsers handle permissions differently; try Chrome or Firefox

For production deployment with Gunicorn:
```
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

[MIT License](LICENSE)

## Troubleshooting

If you encounter any issues:

1. Check that all required dependencies are installed
2. Ensure the model files are correctly downloaded
3. Verify that the audio file format is supported
4. Check browser console for any JavaScript errors
5. For permission issues, visit the test permissions page at `http://localhost:5000/test-permissions`
6. If you encounter prediction errors with the water level model, you can reset it by visiting `http://localhost:5000/reset-flood-model`
