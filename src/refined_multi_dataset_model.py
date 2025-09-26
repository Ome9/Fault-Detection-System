#!/usr/bin/env python3
"""
Refined Multi-Dataset Anomaly Detection Model
Based on the successful optimized model structure with improved multi-dataset handling
Target: Restore 63.4% F1-score performance while maintaining multi-dataset capability
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RefinedFeatureExtractor:
    """Refined 16-feature extractor based on optimized model success"""
    
    def __init__(self, sampling_rate=20000):
        self.fs = sampling_rate
        
    def extract_features(self, signal_data):
        """Extract 16 optimized features similar to successful optimized model"""
        
        # Ensure clean 1D signal
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()
        
        # Remove NaN/inf and ensure minimum length
        signal_data = signal_data[np.isfinite(signal_data)]
        if len(signal_data) < 100:
            signal_data = np.pad(signal_data, (0, max(0, 100 - len(signal_data))), mode='constant')
        
        if len(signal_data) == 0:
            return np.zeros(16, dtype=np.float32)
        
        # Core time domain features (8) - proven effective
        rms = np.sqrt(np.mean(signal_data**2))
        rms = max(rms, 1e-10)  # Prevent division by zero
        
        peak = np.max(np.abs(signal_data))
        crest_factor = peak / rms
        
        # Statistical moments - key fault indicators
        kurtosis = stats.kurtosis(signal_data) if len(signal_data) > 3 else 0.0
        skewness = stats.skew(signal_data) if len(signal_data) > 2 else 0.0
        std_dev = np.std(signal_data)
        mean_abs = np.mean(np.abs(signal_data))
        peak_to_peak = np.max(signal_data) - np.min(signal_data)
        
        # Advanced shape factors (5) - fault sensitivity
        sqrt_mean = np.mean(np.sqrt(np.abs(signal_data)))
        sqrt_mean = max(sqrt_mean, 1e-10)
        
        clearance_factor = peak / (sqrt_mean**2)
        shape_factor = rms / max(mean_abs, 1e-10)
        impulse_factor = peak / max(mean_abs, 1e-10)
        
        # Envelope analysis for bearing faults
        try:
            from scipy import signal as sig
            analytic_signal = sig.hilbert(signal_data)
            envelope = np.abs(analytic_signal)
            envelope_rms = np.sqrt(np.mean(envelope**2))
        except:
            envelope_rms = rms
        
        # Frequency domain (3) - simplified but effective
        try:
            from scipy.fft import fft, fftfreq
            N = len(signal_data)
            if N > 4:
                fft_vals = fft(signal_data[:N//2*2])  # Ensure even length
                freqs = fftfreq(len(fft_vals), 1/self.fs)[:len(fft_vals)//2]
                power_spectrum = np.abs(fft_vals[:len(fft_vals)//2])**2
                
                # Bearing-specific frequency band
                if len(freqs) > 10:
                    bearing_band = (freqs >= 100) & (freqs <= 2000)
                    bearing_power = np.sum(power_spectrum[bearing_band]) if np.any(bearing_band) else 0
                    total_power = np.sum(power_spectrum)
                    spectral_energy = bearing_power / max(total_power, 1e-10)
                else:
                    spectral_energy = 0.0
            else:
                spectral_energy = 0.0
        except:
            spectral_energy = 0.0
        
        # Combine features (16 total)
        features = np.array([
            rms, peak, crest_factor, kurtosis, skewness, std_dev, mean_abs, peak_to_peak,
            clearance_factor, shape_factor, impulse_factor, envelope_rms,
            spectral_energy, np.mean(signal_data), np.median(signal_data), 
            np.percentile(signal_data, 90) - np.percentile(signal_data, 10)
        ], dtype=np.float32)
        
        # Handle any remaining NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features

class RefinedDataLoader:
    """Refined data loader focusing on quality over quantity"""
    
    def __init__(self):
        self.feature_extractor = RefinedFeatureExtractor()
        
    def load_nasa_data(self, limit_files=None):
        """Load NASA bearing dataset with quality focus"""
        print("📊 Loading NASA dataset...")
        base_path = Path("../1st_test")
        
        if not base_path.exists():
            print(f"❌ NASA dataset not found at {base_path}")
            return np.array([]), np.array([])
        
        features = []
        labels = []
        
        # Get all bearing directories
        bearing_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
        
        for bearing_dir in bearing_dirs[:4]:  # Focus on first 4 bearings for quality
            csv_files = sorted(list(bearing_dir.glob("*.csv")))
            total_files = len(csv_files)
            
            if total_files == 0:
                continue
            
            # Apply limit if specified
            if limit_files:
                csv_files = csv_files[:limit_files]
                total_files = len(csv_files)
            
            # More conservative fault threshold (last 25% for clear fault signals)
            fault_threshold = int(total_files * 0.75)
            
            for i, csv_file in enumerate(csv_files):
                try:
                    # NASA data format: space-separated values, multiple channels
                    data = np.loadtxt(csv_file)
                    
                    # Handle both 1D and multi-column data
                    if data.ndim > 1:
                        # Process first two columns (best quality sensors)
                        for col_idx in range(min(2, data.shape[1])):
                            signal = data[:, col_idx]
                            if len(signal) > 2048:  # Ensure sufficient data
                                # Create multiple segments for better representation
                                segment_size = 4096
                                step_size = segment_size // 2  # 50% overlap
                                
                                for start_idx in range(0, len(signal) - segment_size + 1, step_size):
                                    segment = signal[start_idx:start_idx + segment_size]
                                    features.append(self.feature_extractor.extract_features(segment))
                                    labels.append(1 if i >= fault_threshold else 0)
                    else:
                        # 1D data
                        signal = data
                        if len(signal) > 2048:
                            segment_size = 4096
                            step_size = segment_size // 2
                            
                            for start_idx in range(0, len(signal) - segment_size + 1, step_size):
                                segment = signal[start_idx:start_idx + segment_size]
                                features.append(self.feature_extractor.extract_features(segment))
                                labels.append(1 if i >= fault_threshold else 0)
                                
                except Exception as e:
                    print(f"⚠️ Error loading {csv_file}: {e}")
                    continue
        
        print(f"✅ NASA: {len(features)} samples loaded")
        return np.array(features), np.array(labels)
    
    def load_cwru_data(self, limit_files=10):
        """Load CWRU dataset with controlled size"""
        print("📊 Loading CWRU dataset...")
        base_path = Path("../CWRU_Dataset")
        
        if not base_path.exists():
            print(f"❌ CWRU dataset not found at {base_path}")
            return np.array([]), np.array([])
        
        features = []
        labels = []
        
        # Process CSV files with limit
        csv_files = list(base_path.glob("*.csv"))[:limit_files]
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        data = df[col].values
                        if len(data) > 2048:
                            # Create segments with good overlap
                            segment_size = 3072
                            step_size = segment_size // 2
                            
                            for start_idx in range(0, len(data) - segment_size + 1, step_size):
                                segment = data[start_idx:start_idx + segment_size]
                                features.append(self.feature_extractor.extract_features(segment))
                                
                                # Determine label based on filename patterns
                                if any(fault in csv_file.name.lower() for fault in ['fault', 'defect', 'damage', 'ball', 'inner', 'outer']):
                                    labels.append(1)
                                else:
                                    labels.append(0)
                                    
            except Exception as e:
                print(f"⚠️ Error loading {csv_file}: {e}")
                continue
        
        print(f"✅ CWRU: {len(features)} samples loaded")
        return np.array(features), np.array(labels)
    
    def load_hust_data(self, limit_files=5):
        """Load HUST dataset with size control"""
        print("📊 Loading HUST dataset...")
        base_path = Path("../HUST_Dataset")
        
        if not base_path.exists():
            print(f"❌ HUST dataset not found at {base_path}")
            return np.array([]), np.array([])
        
        features = []
        labels = []
        
        # Process files with limit
        file_list = list(base_path.rglob("*.csv"))[:limit_files]
        
        # Look specifically in the HUST bearing dataset subdirectory
        hust_bearing_path = base_path / "HUST bearing dataset"
        if hust_bearing_path.exists():
            mat_files = list(hust_bearing_path.glob("*.mat"))[:limit_files]
            
            for mat_file in mat_files:
                try:
                    import scipy.io as sio
                    data = sio.loadmat(mat_file)
                    
                    # HUST files typically have 'data' key or first non-metadata key
                    signal = None
                    if 'data' in data:
                        signal = data['data'].flatten()
                    else:
                        # Fallback to first non-metadata key
                        data_keys = [k for k in data.keys() if not k.startswith('__')]
                        if data_keys:
                            signal = data[data_keys[0]].flatten()
                    
                    if signal is not None and len(signal) > 1024:
                        # Smaller segments for HUST
                        segment_size = 2048
                        step_size = segment_size // 2
                        
                        for start_idx in range(0, len(signal) - segment_size + 1, step_size):
                            segment = signal[start_idx:start_idx + segment_size]
                            features.append(self.feature_extractor.extract_features(segment))
                            
                            # Simple label assignment based on filename
                            if any(fault_indicator in mat_file.name.lower() 
                                  for fault_indicator in ['fault', 'defect', 'damage', '6']):
                                labels.append(1)
                            else:
                                labels.append(0)
                                
                except Exception as e:
                    print(f"⚠️ Error loading {mat_file}: {e}")
                    continue
        
        print(f"✅ HUST: {len(features)} samples loaded")
        return np.array(features), np.array(labels)

def create_refined_model(input_shape=(16,)):
    """Create refined autoencoder based on successful optimized model"""
    
    # Input layer
    input_layer = keras.Input(shape=input_shape, name='input')
    
    # Encoder - similar to optimized model structure
    x = keras.layers.Dense(48, activation='relu', 
                          kernel_regularizer=keras.regularizers.l2(0.0005),
                          name='encoder_1')(input_layer)
    x = keras.layers.BatchNormalization(name='bn_1')(x)
    x = keras.layers.Dropout(0.1, name='dropout_1')(x)
    
    x = keras.layers.Dense(24, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.0005),
                          name='encoder_2')(x)
    x = keras.layers.BatchNormalization(name='bn_2')(x)
    x = keras.layers.Dropout(0.05, name='dropout_2')(x)
    
    # Bottleneck
    encoded = keras.layers.Dense(12, activation='relu',
                                name='bottleneck')(x)
    
    # Decoder
    x = keras.layers.Dense(24, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.0005),
                          name='decoder_1')(encoded)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = keras.layers.Dropout(0.05, name='dropout_3')(x)
    
    x = keras.layers.Dense(48, activation='relu',
                          kernel_regularizer=keras.regularizers.l2(0.0005),
                          name='decoder_2')(x)
    x = keras.layers.BatchNormalization(name='bn_4')(x)
    x = keras.layers.Dropout(0.1, name='dropout_4')(x)
    
    # Output layer
    output_layer = keras.layers.Dense(input_shape[0], activation='linear',
                                     name='output')(x)
    
    # Create model
    model = keras.Model(input_layer, output_layer, name='refined_autoencoder')
    
    # Compile with proven settings
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def find_optimal_threshold(model, X_val, y_val):
    """Find optimal threshold for anomaly detection"""
    
    # Get reconstruction errors
    X_pred = model.predict(X_val, verbose=0)
    errors = np.mean(np.square(X_val - X_pred), axis=1)
    
    # Try different thresholds
    thresholds = np.linspace(np.min(errors), np.max(errors), 200)
    best_threshold = 0
    best_f1 = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (errors > threshold).astype(int)
        
        # Skip if all predictions are the same
        if len(np.unique(y_pred)) == 1:
            continue
            
        # Calculate metrics
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        accuracy = accuracy_score(y_val, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'f1': f1
            }
    
    print(f"🎯 Optimal threshold: {best_threshold:.6f}")
    print(f"   Best F1-score: {best_metrics['f1']:.4f}")
    print(f"   Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   Best Precision: {best_metrics['precision']:.4f}")
    print(f"   Best Recall: {best_metrics['recall']:.4f}")
    
    return best_threshold, best_metrics

def main():
    print("🚀 Starting Refined Multi-Dataset Model Training...")
    print("="*80)
    
    # Load data with controlled sizes
    data_loader = RefinedDataLoader()
    
    nasa_features, nasa_labels = data_loader.load_nasa_data(limit_files=30)
    cwru_features, cwru_labels = data_loader.load_cwru_data(limit_files=8)
    hust_features, hust_labels = data_loader.load_hust_data(limit_files=4)
    
    # Combine datasets
    all_features = []
    all_labels = []
    
    if len(nasa_features) > 0:
        all_features.append(nasa_features)
        all_labels.append(nasa_labels)
    
    if len(cwru_features) > 0:
        all_features.append(cwru_features)
        all_labels.append(cwru_labels)
    
    if len(hust_features) > 0:
        all_features.append(hust_features)
        all_labels.append(hust_labels)
    
    if not all_features:
        print("❌ No data loaded! Please check dataset paths.")
        return
    
    # Combine data
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Normal: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"   Fault: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n🔄 Data Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Create and train model
    print(f"\n🏗️ Building Refined Model...")
    model = create_refined_model(input_shape=(X_train.shape[1],))
    
    print(f"📊 Model Parameters: {model.count_params():,}")
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_refined_model.h5',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n🎯 Training Refined Model...")
    history = model.fit(
        X_train, X_train,  # Autoencoder: input = output
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Find optimal threshold
    print(f"\n🎯 Finding optimal threshold...")
    threshold, best_metrics = find_optimal_threshold(model, X_val, y_val)
    
    # Evaluate on test set
    print(f"\n📈 Evaluating Refined Multi-Dataset Model...")
    X_test_pred = model.predict(X_test, verbose=0)
    test_errors = np.mean(np.square(X_test - X_test_pred), axis=1)
    y_pred = (test_errors > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Calculate per-class accuracy
    normal_mask = y_test == 0
    fault_mask = y_test == 1
    
    normal_accuracy = accuracy_score(y_test[normal_mask], y_pred[normal_mask]) if np.sum(normal_mask) > 0 else 0
    fault_accuracy = accuracy_score(y_test[fault_mask], y_pred[fault_mask]) if np.sum(fault_mask) > 0 else 0
    
    print(f"\n🏆 REFINED MULTI-DATASET MODEL RESULTS:")
    print(f"   Overall Accuracy: {accuracy*100:.1f}%")
    print(f"   Precision: {precision*100:.1f}%")
    print(f"   Recall (Fault Detection): {recall*100:.1f}%")
    print(f"   Specificity (Normal Detection): {normal_accuracy*100:.1f}%")
    print(f"   F1-Score: {f1*100:.1f}%")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Normal  Fault")
    print(f"   Actual Normal   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"          Fault     {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    # Error analysis
    normal_errors = test_errors[y_test == 0]
    fault_errors = test_errors[y_test == 1]
    
    print(f"\n🔍 Performance Analysis:")
    if len(normal_errors) > 0:
        print(f"   Normal samples error: {np.mean(normal_errors):.6f}")
    if len(fault_errors) > 0:
        print(f"   Fault samples error: {np.mean(fault_errors):.6f}")
        if len(normal_errors) > 0:
            print(f"   Error separation: {np.mean(fault_errors)/np.mean(normal_errors):.2f}x")
    
    # Save model for deployment
    print(f"\n💾 Saving Refined Multi-Dataset Model...")
    
    # Create deployment directory
    deployment_dir = Path("refined_deployment")
    deployment_dir.mkdir(exist_ok=True)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = deployment_dir / "refined_model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    # Generate C header
    c_header_path = deployment_dir / "refined_model_data.h"
    with open(c_header_path, 'w') as f:
        f.write('#ifndef REFINED_MODEL_DATA_H\n')
        f.write('#define REFINED_MODEL_DATA_H\n\n')
        f.write('#include <stdint.h>\n\n')
        f.write(f'const unsigned int refined_model_len = {len(tflite_model)};\n')
        f.write('const unsigned char refined_model_data[] = {\n')
        
        # Write model data
        for i, byte in enumerate(tflite_model):
            if i % 12 == 0:
                f.write('\n  ')
            f.write(f'0x{byte:02x}, ')
        
        f.write('\n};\n\n')
        f.write('#endif  // REFINED_MODEL_DATA_H\n')
    
    model_size_kb = len(tflite_model) / 1024
    print(f"✅ Refined model size: {model_size_kb:.2f} KB")
    print(f"✅ Refined C header saved: {c_header_path}")
    
    # Performance evolution comparison
    print(f"\n🎊 PERFORMANCE EVOLUTION:")
    print(f"   Original Model:        F1=10.7%, Accuracy=N/A")
    print(f"   Improved Model:        F1=45.3%, Accuracy=N/A")
    print(f"   Optimized Model:       F1=63.4%, Accuracy=N/A  ⭐ TARGET")
    print(f"   Enhanced Model:        F1=46.1%, Accuracy=30.0%")
    print(f"   Improved Model:        F1=34.0%, Accuracy=29.8%")
    print(f"   🏆 REFINED Model:      F1={f1*100:.1f}%, Accuracy={accuracy*100:.1f}%")
    
    # Success indicator
    if f1 >= 0.60:  # Close to optimized model performance
        print(f"\n🎉 SUCCESS: Refined model achieved target performance!")
        print(f"   ✅ F1-Score {f1*100:.1f}% ≥ 60% target")
    elif f1 >= 0.50:
        print(f"\n🎯 GOOD: Refined model shows strong improvement!")
        print(f"   ✅ F1-Score {f1*100:.1f}% ≥ 50% threshold")
    else:
        print(f"\n📈 PROGRESS: Refined model needs further optimization")
        print(f"   ⚠️ F1-Score {f1*100:.1f}% < 50% threshold")
    
    print(f"\n🎉 REFINED MULTI-DATASET TRAINING COMPLETE!")
    print("="*80)
    print(f"📁 Output directory: {deployment_dir}/")
    print(f"🔥 Model size: {model_size_kb:.1f} KB")
    print(f"⚡ Features: 16 optimized indicators")
    print(f"🧠 Parameters: {model.count_params():,}")
    print(f"📊 F1-Score: {f1*100:.1f}%")
    print(f"🎯 Fault Detection: {recall*100:.1f}%")
    print(f"🛡️ Normal Detection: {normal_accuracy*100:.1f}%")
    print(f"⚖️ Accuracy: {accuracy*100:.1f}%")

if __name__ == "__main__":
    main()