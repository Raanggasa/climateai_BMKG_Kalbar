import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Any
import logging
from datetime import datetime, timezone
import pickle
import os

class WeatherPredictor:
    """Base class for weather prediction models"""
    
    def __init__(self, sequence_length: int = 72, features: int = 6, model_name: str = "base"):
        self.sequence_length = sequence_length
        self.features = features
        self.model_name = model_name
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def prepare_sequences(self, data: np.ndarray, target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training/prediction"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, target_col])
            
        return np.array(X), np.array(y)
    
    def save_model(self, model_dir: str = "models"):
        """Save trained model and scaler"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Save TensorFlow model
        model_path = os.path.join(model_dir, f"{self.model_name}_model.h5")
        self.model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        logging.info(f"Model {self.model_name} saved to {model_path}")
        
    def load_model(self, model_dir: str = "models"):
        """Load trained model and scaler"""
        model_path = os.path.join(model_dir, f"{self.model_name}_model.h5")
        scaler_path = os.path.join(model_dir, f"{self.model_name}_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            self.is_trained = True
            logging.info(f"Model {self.model_name} loaded successfully")
            return True
        else:
            logging.warning(f"Model files not found for {self.model_name}")
            return False

class WaterLevelPredictor(WeatherPredictor):
    """LSTM model for water level prediction (flood risk)"""
    
    def __init__(self, sequence_length: int = 168, features: int = 6):
        super().__init__(sequence_length, features, "water_level")
        self._build_model()
    
    def _build_model(self):
        """Build LSTM architecture for water level prediction"""
        self.model = Sequential([
            LSTM(128, return_sequences=True, 
                 input_shape=(self.sequence_length, self.features)),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logging.info("Water Level Predictor model built successfully")
    
    def create_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to water level prediction"""
        features = pd.DataFrame({
            'waterlevel': weather_data['waterlevel'],
            'rain': weather_data['rain'],
            'pressure': weather_data['pressure'],
            'temp': weather_data['temp'],
            'rh': weather_data['rh'],
            'windspeed': weather_data['windspeed']
        })
        
        return features.fillna(method='bfill').fillna(method='ffill')
    
    def train(self, weather_data: pd.DataFrame, epochs: int = 50, validation_split: float = 0.2):
        """Train the water level prediction model"""
        try:
            # Create features
            features_df = self.create_features(weather_data)
            
            if len(features_df) < self.sequence_length + 10:
                raise ValueError(f"Insufficient data. Need at least {self.sequence_length + 10} records")
            
            # Scale features
            scaled_data = self.scaler.fit_transform(features_df)
            
            # Prepare sequences (target is water level - column 0)
            X, y = self.prepare_sequences(scaled_data, target_col=0)
            
            if len(X) == 0:
                raise ValueError("No valid sequences generated")
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Define callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10)
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            logging.info("Water level model training completed")
            
            return history
            
        except Exception as e:
            logging.error(f"Error training water level model: {str(e)}")
            raise
    
    def predict_flood_risk(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict water level and flood risk"""
        if not self.is_trained:
            raise ValueError("Model not trained. Train the model first.")
        
        try:
            # Create features
            features_df = self.create_features(recent_data)
            
            if len(features_df) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
            # Scale features
            scaled_features = self.scaler.transform(features_df)
            
            # Get the last sequence for prediction
            sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            prediction_scaled = self.model.predict(sequence, verbose=0)
            
            # Inverse transform prediction (only for water level column)
            dummy_array = np.zeros((1, self.features))
            dummy_array[0, 0] = prediction_scaled[0, 0]
            prediction_original = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            # Determine flood risk level
            current_level = recent_data['waterlevel'].iloc[-1]
            predicted_level = prediction_original
            
            # Define flood risk thresholds (in meters)
            if predicted_level > 3.0:
                risk_level = "High Risk"
                risk_color = "red"
            elif predicted_level > 2.5:
                risk_level = "Medium Risk"
                risk_color = "orange"
            elif predicted_level > 2.0:
                risk_level = "Low Risk"
                risk_color = "yellow"
            else:
                risk_level = "Normal"
                risk_color = "green"
            
            return {
                "predicted_water_level_m": float(predicted_level),
                "current_water_level_m": float(current_level),
                "level_change_m": float(predicted_level - current_level),
                "flood_risk_level": risk_level,
                "risk_color": risk_color,
                "confidence": 0.85  # Placeholder - would calculate based on model uncertainty
            }
            
        except Exception as e:
            logging.error(f"Error predicting flood risk: {str(e)}")
            raise

class WindPatternPredictor(WeatherPredictor):
    """GRU model for wind pattern prediction (maritime weather)"""
    
    def __init__(self, sequence_length: int = 72, features: int = 8):
        super().__init__(sequence_length, features, "wind_pattern")
        self._build_model()
    
    def _build_model(self):
        """Build bidirectional GRU architecture for wind prediction"""
        self.model = Sequential([
            Bidirectional(
                GRU(64, return_sequences=True),
                input_shape=(self.sequence_length, self.features)
            ),
            Dropout(0.3),
            Bidirectional(
                GRU(32, return_sequences=False)
            ),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(2, activation='linear')  # Wind speed and direction
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.002),
            loss='mse',
            metrics=['mae']
        )
        
        logging.info("Wind Pattern Predictor model built successfully")
    
    def create_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Create wind-specific features"""
        # Calculate wind components
        wind_u = weather_data['windspeed'] * np.cos(np.radians(weather_data['winddir']))
        wind_v = weather_data['windspeed'] * np.sin(np.radians(weather_data['winddir']))
        
        # Add pressure gradients and temperature differentials
        pressure_gradient = weather_data['pressure'].diff()
        temp_gradient = weather_data['temp'].diff()
        
        features = pd.DataFrame({
            'wind_u': wind_u,
            'wind_v': wind_v,
            'pressure': weather_data['pressure'],
            'pressure_gradient': pressure_gradient,
            'temp': weather_data['temp'],
            'temp_gradient': temp_gradient,
            'rh': weather_data['rh'],
            'hour_sin': np.sin(2 * np.pi * weather_data.index.hour / 24) if hasattr(weather_data.index, 'hour') else 0
        })
        
        return features.fillna(method='bfill').fillna(method='ffill')
    
    def predict_wind_conditions(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict wind speed and direction for maritime safety"""
        if not self.is_trained:
            raise ValueError("Model not trained. Train the model first.")
        
        try:
            # Create features
            features_df = self.create_features(recent_data)
            
            if len(features_df) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
            # Scale features
            scaled_features = self.scaler.transform(features_df)
            
            # Get the last sequence for prediction
            sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            
            # Extract wind speed and direction from prediction
            wind_u_pred = prediction[0][0]
            wind_v_pred = prediction[0][1]
            
            # Convert back to speed and direction
            wind_speed = np.sqrt(wind_u_pred**2 + wind_v_pred**2)
            wind_direction = np.degrees(np.arctan2(wind_v_pred, wind_u_pred)) % 360
            
            # Determine maritime safety level
            current_speed = recent_data['windspeed'].iloc[-1]
            
            if wind_speed > 15:
                safety_level = "Dangerous"
                safety_color = "red"
            elif wind_speed > 10:
                safety_level = "Caution"
                safety_color = "orange"
            elif wind_speed > 5:
                safety_level = "Moderate"
                safety_color = "yellow"
            else:
                safety_level = "Safe"
                safety_color = "green"
            
            return {
                "predicted_wind_speed_ms": float(wind_speed),
                "predicted_wind_direction_deg": float(wind_direction),
                "current_wind_speed_ms": float(current_speed),
                "speed_change_ms": float(wind_speed - current_speed),
                "maritime_safety_level": safety_level,
                "safety_color": safety_color,
                "confidence": 0.82
            }
            
        except Exception as e:
            logging.error(f"Error predicting wind conditions: {str(e)}")
            raise

class RainfallPredictor(WeatherPredictor):
    """CNN-LSTM hybrid for rainfall probability prediction"""
    
    def __init__(self, sequence_length: int = 96, features: int = 10):
        super().__init__(sequence_length, features, "rainfall")
        self._build_model()
    
    def _build_model(self):
        """Build CNN-LSTM hybrid architecture"""
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu',
                   input_shape=(self.sequence_length, self.features)),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            LSTM(50, return_sequences=True),
            Dropout(0.3),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')  # Probability output
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logging.info("Rainfall Predictor model built successfully")
    
    def create_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Create rainfall-specific features"""
        # Calculate dewpoint using Magnus formula
        def calculate_dewpoint(temp, humidity):
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100.0)
            return (b * alpha) / (a - alpha)
        
        dewpoint = calculate_dewpoint(weather_data['temp'], weather_data['rh'])
        
        features = pd.DataFrame({
            'temp': weather_data['temp'],
            'rh': weather_data['rh'],
            'pressure': weather_data['pressure'],
            'windspeed': weather_data['windspeed'],
            'dewpoint': dewpoint,
            'pressure_trend': weather_data['pressure'].rolling(6).mean().diff(),
            'humidity_trend': weather_data['rh'].rolling(3).mean().diff(),
            'temp_dewpoint_diff': weather_data['temp'] - dewpoint,
            'hour_sin': np.sin(2 * np.pi * weather_data.index.hour / 24) if hasattr(weather_data.index, 'hour') else 0,
            'day_sin': np.sin(2 * np.pi * weather_data.index.dayofyear / 365) if hasattr(weather_data.index, 'dayofyear') else 0
        })
        
        return features.fillna(method='bfill').fillna(method='ffill')
    
    def predict_rainfall_probability(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict probability of rainfall occurrence"""
        if not self.is_trained:
            raise ValueError("Model not trained. Train the model first.")
        
        try:
            # Create features
            features_df = self.create_features(recent_data)
            
            if len(features_df) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
            # Scale features
            scaled_features = self.scaler.transform(features_df)
            
            # Get the last sequence for prediction
            sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            probability = self.model.predict(sequence, verbose=0)[0][0]
            
            # Categorize probability
            if probability < 0.2:
                category = "Very Low"
                color = "green"
            elif probability < 0.4:
                category = "Low"
                color = "lightgreen"
            elif probability < 0.6:
                category = "Moderate"
                color = "yellow"
            elif probability < 0.8:
                category = "High"
                color = "orange"
            else:
                category = "Very High"
                color = "red"
            
            return {
                "rainfall_probability_percent": float(probability * 100),
                "rainfall_category": category,
                "category_color": color,
                "current_conditions": {
                    "humidity": float(recent_data['rh'].iloc[-1]),
                    "temperature": float(recent_data['temp'].iloc[-1]),
                    "pressure": float(recent_data['pressure'].iloc[-1])
                },
                "confidence": 0.78
            }
            
        except Exception as e:
            logging.error(f"Error predicting rainfall: {str(e)}")
            raise

class TemperatureHumidityPredictor(WeatherPredictor):
    """Multi-output LSTM for temperature and humidity prediction"""
    
    def __init__(self, sequence_length: int = 120, features: int = 12):
        super().__init__(sequence_length, features, "temp_humidity")
        self._build_model()
    
    def _build_model(self):
        """Build multi-output LSTM architecture"""
        # Shared LSTM layers
        input_layer = Input(shape=(self.sequence_length, self.features))
        
        lstm1 = LSTM(100, return_sequences=True)(input_layer)
        dropout1 = Dropout(0.3)(lstm1)
        
        lstm2 = LSTM(75, return_sequences=True)(dropout1)
        dropout2 = Dropout(0.3)(lstm2)
        
        lstm3 = LSTM(50, return_sequences=False)(dropout2)
        dropout3 = Dropout(0.2)(lstm3)
        
        # Separate output heads
        temp_dense = Dense(32, activation='relu', name='temp_dense')(dropout3)
        temp_output = Dense(1, activation='linear', name='temperature')(temp_dense)
        
        humidity_dense = Dense(32, activation='relu', name='humidity_dense')(dropout3)
        humidity_output = Dense(1, activation='linear', name='humidity')(humidity_dense)
        
        self.model = Model(
            inputs=input_layer,
            outputs=[temp_output, humidity_output]
        )
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'temperature': 'mse',
                'humidity': 'mse'
            },
            loss_weights={
                'temperature': 1.0,
                'humidity': 0.8
            },
            metrics={
                'temperature': ['mae'],
                'humidity': ['mae']
            }
        )
        
        logging.info("Temperature-Humidity Predictor model built successfully")
    
    def create_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for temperature and humidity"""
        features = pd.DataFrame({
            'temp': weather_data['temp'],
            'rh': weather_data['rh'],
            'pressure': weather_data['pressure'],
            'windspeed': weather_data['windspeed'],
            'solrad': weather_data.get('solrad', 0),
            'netrad': weather_data.get('netrad', 0),
            'temp_lag1': weather_data['temp'].shift(1),
            'humidity_lag1': weather_data['rh'].shift(1),
            'temp_ma6': weather_data['temp'].rolling(6).mean(),
            'humidity_ma6': weather_data['rh'].rolling(6).mean(),
            'hour_sin': np.sin(2 * np.pi * weather_data.index.hour / 24) if hasattr(weather_data.index, 'hour') else 0,
            'hour_cos': np.cos(2 * np.pi * weather_data.index.hour / 24) if hasattr(weather_data.index, 'hour') else 0
        })
        
        return features.fillna(method='bfill').fillna(method='ffill')
    
    def predict_climate_comfort(self, recent_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict temperature and humidity for urban climate comfort"""
        if not self.is_trained:
            raise ValueError("Model not trained. Train the model first.")
        
        try:
            # Create features
            features_df = self.create_features(recent_data)
            
            if len(features_df) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
            # Scale features
            scaled_features = self.scaler.transform(features_df)
            
            # Get the last sequence for prediction
            sequence = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            predictions = self.model.predict(sequence, verbose=0)
            temp_pred = predictions[0][0][0]
            humidity_pred = predictions[1][0][0]
            
            # Calculate heat index for comfort level
            def calculate_heat_index(temp_c, humidity_percent):
                temp_f = temp_c * 9/5 + 32
                hi = temp_f
                if temp_f >= 80:
                    hi = (-42.379 + 2.04901523*temp_f + 10.14333127*humidity_percent - 
                         0.22475541*temp_f*humidity_percent - 0.00683783*temp_f*temp_f -
                         0.05481717*humidity_percent*humidity_percent + 
                         0.00122874*temp_f*temp_f*humidity_percent + 
                         0.00085282*temp_f*humidity_percent*humidity_percent -
                         0.00000199*temp_f*temp_f*humidity_percent*humidity_percent)
                return (hi - 32) * 5/9  # Convert back to Celsius
            
            heat_index = calculate_heat_index(temp_pred, humidity_pred)
            
            # Determine comfort level
            if heat_index > 32:
                comfort_level = "Dangerous Heat"
                comfort_color = "red"
            elif heat_index > 28:
                comfort_level = "Very Hot"
                comfort_color = "orange"
            elif heat_index > 24:
                comfort_level = "Hot"
                comfort_color = "yellow"
            elif heat_index > 18:
                comfort_level = "Comfortable"
                comfort_color = "green"
            else:
                comfort_level = "Cool"
                comfort_color = "blue"
            
            current_temp = recent_data['temp'].iloc[-1]
            current_humidity = recent_data['rh'].iloc[-1]
            
            return {
                "predicted_temperature_c": float(temp_pred),
                "predicted_humidity_percent": float(humidity_pred),
                "predicted_heat_index_c": float(heat_index),
                "current_temperature_c": float(current_temp),
                "current_humidity_percent": float(current_humidity),
                "temperature_change_c": float(temp_pred - current_temp),
                "humidity_change_percent": float(humidity_pred - current_humidity),
                "comfort_level": comfort_level,
                "comfort_color": comfort_color,
                "confidence": 0.88
            }
            
        except Exception as e:
            logging.error(f"Error predicting temperature/humidity: {str(e)}")
            raise

# Model manager class
class WeatherModelManager:
    """Manager for all weather prediction models"""
    
    def __init__(self):
        self.models = {
            'flood_risk': WaterLevelPredictor(),
            'wind_pattern': WindPatternPredictor(),
            'rainfall': RainfallPredictor(),
            'temp_humidity': TemperatureHumidityPredictor()
        }
        self.training_status = {
            'flood_risk': False,
            'wind_pattern': False,
            'rainfall': False,
            'temp_humidity': False
        }
    
    def check_training_data_sufficiency(self, weather_data: pd.DataFrame) -> Dict[str, bool]:
        """Check if we have sufficient data for training each model"""
        data_requirements = {
            'flood_risk': 168 + 50,   # 7 days + buffer
            'wind_pattern': 72 + 50,  # 3 days + buffer
            'rainfall': 96 + 50,      # 4 days + buffer
            'temp_humidity': 120 + 50 # 5 days + buffer
        }
        
        sufficient_data = {}
        data_count = len(weather_data)
        
        for model_name, required_count in data_requirements.items():
            sufficient_data[model_name] = data_count >= required_count
        
        return sufficient_data
    
    def train_models(self, weather_data: pd.DataFrame) -> Dict[str, Any]:
        """Train all models if sufficient data is available"""
        results = {}
        sufficient_data = self.check_training_data_sufficiency(weather_data)
        
        for model_name, model in self.models.items():
            try:
                if sufficient_data[model_name]:
                    logging.info(f"Training {model_name} model...")
                    
                    # Train model with appropriate epochs based on data size
                    epochs = min(100, max(20, len(weather_data) // 50))
                    history = model.train(weather_data, epochs=epochs)
                    
                    # Save trained model
                    model.save_model()
                    
                    self.training_status[model_name] = True
                    results[model_name] = {
                        'status': 'trained',
                        'epochs': epochs,
                        'data_points': len(weather_data)
                    }
                    
                    logging.info(f"{model_name} model training completed")
                else:
                    results[model_name] = {
                        'status': 'insufficient_data',
                        'required': list(self.check_training_data_sufficiency(weather_data).values())[list(self.models.keys()).index(model_name)],
                        'available': len(weather_data)
                    }
                    
            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def load_trained_models(self) -> Dict[str, bool]:
        """Load all previously trained models"""
        loaded_models = {}
        
        for model_name, model in self.models.items():
            try:
                success = model.load_model()
                loaded_models[model_name] = success
                if success:
                    self.training_status[model_name] = True
            except Exception as e:
                logging.error(f"Error loading {model_name}: {str(e)}")
                loaded_models[model_name] = False
        
        return loaded_models
    
    def get_predictions(self, weather_data: pd.DataFrame) -> Dict[str, Any]:
        """Get predictions from all trained models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            if self.training_status[model_name]:
                try:
                    if model_name == 'flood_risk':
                        predictions[model_name] = model.predict_flood_risk(weather_data)
                    elif model_name == 'wind_pattern':
                        predictions[model_name] = model.predict_wind_conditions(weather_data)
                    elif model_name == 'rainfall':
                        predictions[model_name] = model.predict_rainfall_probability(weather_data)
                    elif model_name == 'temp_humidity':
                        predictions[model_name] = model.predict_climate_comfort(weather_data)
                        
                except Exception as e:
                    logging.error(f"Error getting prediction from {model_name}: {str(e)}")
                    predictions[model_name] = {'error': str(e)}
            else:
                predictions[model_name] = {'error': 'Model not trained'}
        
        return predictions
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            'models_status': self.training_status,
            'total_models': len(self.models),
            'trained_models': sum(self.training_status.values()),
            'models_info': {
                name: {
                    'trained': status,
                    'sequence_length': model.sequence_length,
                    'features': model.features
                }
                for name, (model, status) in zip(self.models.keys(), 
                                                zip(self.models.values(), 
                                                   self.training_status.values()))
            }
        }