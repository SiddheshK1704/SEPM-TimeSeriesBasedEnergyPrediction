from flask import Flask, render_template, request, jsonify, session, redirect
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import bcrypt
import pandas as pd
import numpy as np
import os
import traceback
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', ''),
            database=os.getenv('MYSQL_DATABASE', 'energy_forecast_db')
        )
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/')
    return render_template('dashboard.html')

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': 'Username already exists'}), 400
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, hashed_password.decode('utf-8'))
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Account created successfully'})
    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/signin', methods=['POST'])
def signin():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required'}), 400
    
    conn = get_db_connection()
    if not conn:
        return jsonify({'success': False, 'message': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({'success': True, 'message': 'Login successful', 'username': user['username']})
    
    return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'username': session.get('username')})
    return jsonify({'authenticated': False})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store file info in session
        session['uploaded_file'] = filepath
        
        # Read and return preview
        try:
            df = pd.read_csv(filepath)
            preview = df.head(10).to_dict(orient='records')
            columns = df.columns.tolist()
            
            # Check if the target column exists
            target_col = "Electricity:Facility [kW](Hourly)"
            has_target = target_col in df.columns
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename,
                'preview': preview,
                'columns': columns,
                'has_target': has_target,
                'target_column': target_col if has_target else None,
                'row_count': len(df)
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'}), 500
    
    return jsonify({'success': False, 'message': 'Invalid file type. Please upload CSV'}), 400

@app.route('/api/forecast', methods=['POST'])
def forecast():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    data = request.json
    model_type = data.get('model_type', 'arima')
    forecast_hours = int(data.get('forecast_hours', 24))
    
    filepath = session.get('uploaded_file')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'success': False, 'message': 'No data file uploaded. Please upload a CSV first.'}), 400
    
    try:
        # Read the uploaded data
        df = pd.read_csv(filepath)
        target_col = "Electricity:Facility [kW](Hourly)"
        
        if target_col not in df.columns:
            return jsonify({'success': False, 'message': f'Target column "{target_col}" not found in the uploaded file'}), 400
        
        series = df[target_col].dropna()
        
        if len(series) < 50:
            return jsonify({'success': False, 'message': 'Not enough data for forecasting. Need at least 50 data points.'}), 400
        
        # Perform forecasting based on model type
        if model_type == 'arima':
            predictions = forecast_arima(series, forecast_hours)
        elif model_type == 'rnn':
            predictions = forecast_rnn(series, forecast_hours)
        elif model_type == 'lstm':
            predictions = forecast_lstm(series, forecast_hours)
        else:
            return jsonify({'success': False, 'message': 'Invalid model type'}), 400
        
        # Generate timestamps for predictions
        last_date = datetime.now()
        timestamps = [(last_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:00:00') for i in range(1, forecast_hours + 1)]
        
        # Calculate metrics
        actual_values = series.tail(min(forecast_hours, len(series) // 5)).values
        pred_values = predictions[:len(actual_values)]
        
        if len(actual_values) > 0 and len(pred_values) > 0:
            mae = np.mean(np.abs(actual_values - pred_values))
            rmse = np.sqrt(np.mean((actual_values - pred_values) ** 2))
            mape = np.mean(np.abs((actual_values - pred_values) / (actual_values + 1e-10))) * 100
        else:
            mae = rmse = mape = 0
        
        # Get historical data for chart
        historical = {
            'timestamps': df['Date/Time'].tail(100).tolist() if 'Date/Time' in df.columns else list(range(100)),
            'values': series.tail(100).tolist()
        }
        
        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'timestamps': timestamps,
            'metrics': {
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2)
            },
            'historical': historical
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Forecasting error: {str(e)}'}), 500

def forecast_arima(series, steps):
    """Simple ARIMA forecasting using last values and trend"""
    from statsmodels.tsa.arima.model import ARIMA
    
    # Use a simple ARIMA model
    model = ARIMA(series.values, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def forecast_rnn(series, steps):
    """Simple forecasting using moving average and trend for demo"""
    # If actual RNN model is available, use it
    try:
        from tensorflow.keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        
        model_path = 'models/rnn_model.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
            
            # Use last 168 hours for prediction
            time_steps = 168
            if len(scaled_data) >= time_steps:
                last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
                predictions = []
                current_seq = last_sequence.copy()
                
                for _ in range(steps):
                    pred_scaled = model.predict(current_seq, verbose=0)
                    predictions.append(pred_scaled[0, 0])
                    # Update sequence
                    new_seq = np.append(current_seq[0, 1:, :], pred_scaled.reshape(1, 1, 1), axis=0)
                    current_seq = new_seq.reshape(1, time_steps, 1)
                
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                return predictions.flatten()
    except:
        pass
    
    # Fallback: Simple exponential smoothing
    alpha = 0.3
    smoothed = [series.values[0]]
    for val in series.values[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    
    last_value = smoothed[-1]
    trend = np.mean(np.diff(smoothed[-50:])) if len(smoothed) > 50 else 0
    
    predictions = [last_value + trend * (i + 1) for i in range(steps)]
    return np.array(predictions)

def forecast_lstm(series, steps):
    """Simple forecasting using moving average and trend for demo"""
    # If actual LSTM model is available, use it
    try:
        from tensorflow.keras.models import load_model
        from sklearn.preprocessing import MinMaxScaler
        
        model_path = 'models/electricity_lstm_model.h5'
        if os.path.exists(model_path):
            model = load_model(model_path)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
            
            # Use last 168 hours for prediction
            time_steps = 168
            if len(scaled_data) >= time_steps:
                last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
                predictions = []
                current_seq = last_sequence.copy()
                
                for _ in range(steps):
                    pred_scaled = model.predict(current_seq, verbose=0)
                    predictions.append(pred_scaled[0, 0])
                    # Update sequence
                    new_seq = np.append(current_seq[0, 1:, :], pred_scaled.reshape(1, 1, 1), axis=0)
                    current_seq = new_seq.reshape(1, time_steps, 1)
                
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                return predictions.flatten()
    except:
        pass
    
    # Fallback: Simple linear regression
    x = np.arange(len(series))
    z = np.polyfit(x, series.values, 1)
    p = np.poly1d(z)
    
    predictions = [p(len(series) + i) for i in range(steps)]
    return np.array(predictions)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    data = request.json
    consumption = data.get('consumption', 0)
    
    recommendations = generate_recommendations(consumption)
    return jsonify({'success': True, 'recommendations': recommendations})

def generate_recommendations(consumption):
    """Generate energy saving recommendations based on consumption"""
    recommendations = []
    
    if consumption > 100:
        recommendations.append({
            'title': 'High Consumption Alert',
            'description': 'Your energy consumption is very high. Consider checking for appliances that might be running unnecessarily.',
            'priority': 'high',
            'savings': 'Up to 30% reduction possible'
        })
    
    recommendations.extend([
        {
            'title': 'Smart Scheduling',
            'description': 'Run heavy appliances like washing machines, dishwashers during off-peak hours (10 PM - 6 AM).',
            'priority': 'medium',
            'savings': '15-20% reduction'
        },
        {
            'title': 'HVAC Optimization',
            'description': 'Set thermostat to 68°F (20°C) in winter and 78°F (26°C) in summer for optimal savings.',
            'priority': 'high',
            'savings': '10-15% reduction'
        },
        {
            'title': 'Lighting Efficiency',
            'description': 'Switch to LED bulbs and utilize natural light during daytime hours.',
            'priority': 'low',
            'savings': '5-10% reduction'
        },
        {
            'title': 'Standby Power',
            'description': 'Unplug electronics when not in use. Many devices consume power even when turned off.',
            'priority': 'medium',
            'savings': '5-8% reduction'
        }
    ])
    
    if consumption < 50:
        recommendations.append({
            'title': 'Great Job!',
            'description': 'Your consumption is below average. Keep up the good habits!',
            'priority': 'low',
            'savings': 'Maintain current usage'
        })
    
    return recommendations

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)