#!/usr/bin/env python3
"""
Fixed Complete TinyML Deployment Pipeline for NASA Bearing Fault Detection
This script creates a comprehensive deployment package for STM32 microcontrollers

Author: AI Assistant
Date: 2025
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TinyMLFeatureExtractor:
    """Extract optimized features for TinyML deployment."""
    
    def __init__(self, sampling_rate=20000):
        self.sampling_rate = sampling_rate
        
    def extract_features(self, signal):
        """Extract 8 optimized features for TinyML."""
        features = {}
        
        # 1. RMS (Root Mean Square) - overall energy
        features['rms'] = np.sqrt(np.mean(signal**2))
        
        # 2. Peak amplitude
        features['peak'] = np.max(np.abs(signal))
        
        # 3. Crest Factor (Peak/RMS) - impulsiveness indicator
        features['crest_factor'] = features['peak'] / (features['rms'] + 1e-8)
        
        # 4. Kurtosis - statistical impulsiveness
        features['kurtosis'] = self._calculate_kurtosis(signal)
        
        # 5. Envelope peak (Hilbert transform approximation)
        features['envelope_peak'] = self._envelope_peak(signal)
        
        # 6. High frequency power (simplified)
        features['high_freq_power'] = self._high_freq_power(signal)
        
        # 7. Bearing frequency power (approximated)
        features['bearing_freq_power'] = self._bearing_freq_power(signal)
        
        # 8. Spectral kurtosis (simplified)
        features['spectral_kurtosis'] = self._spectral_kurtosis(signal)
        
        return np.array(list(features.values()))
    
    def _calculate_kurtosis(self, signal):
        """Calculate kurtosis with numerical stability."""
        mean = np.mean(signal)
        std = np.std(signal) + 1e-8
        return np.mean(((signal - mean) / std) ** 4) - 3
    
    def _envelope_peak(self, signal):
        """Approximate envelope peak using moving average."""
        # Simple envelope approximation
        abs_signal = np.abs(signal)
        window_size = len(signal) // 20
        envelope = np.convolve(abs_signal, np.ones(window_size)/window_size, mode='same')
        return np.max(envelope)
    
    def _high_freq_power(self, signal):
        """Calculate high frequency power (simplified FFT)."""
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        power_spectrum = np.abs(fft)**2
        
        # High frequency band (>5kHz)
        high_freq_mask = np.abs(freqs) > 5000
        return np.sum(power_spectrum[high_freq_mask])
    
    def _bearing_freq_power(self, signal):
        """Bearing frequency power (simplified)."""
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        power_spectrum = np.abs(fft)**2
        
        # Typical bearing fault frequencies (100-2000 Hz)
        bearing_freq_mask = (np.abs(freqs) > 100) & (np.abs(freqs) < 2000)
        return np.sum(power_spectrum[bearing_freq_mask])
    
    def _spectral_kurtosis(self, signal):
        """Simplified spectral kurtosis."""
        fft = np.fft.fft(signal)
        power_spectrum = np.abs(fft)**2
        return self._calculate_kurtosis(power_spectrum)

class TinyMLAutoencoder:
    """Ultra-compact autoencoder for TinyML deployment."""
    
    def __init__(self, input_dim=8):
        self.input_dim = input_dim
        self.autoencoder = None
        self.scaler = MinMaxScaler()
        self.threshold = None
        
    def build_model(self):
        """Build optimized autoencoder architecture."""
        # Simple but effective architecture for TinyML
        inputs = keras.Input(shape=(self.input_dim,), name='input')
        
        # Encoder
        encoded = layers.Dense(16, activation='relu', name='encode_1')(inputs)
        encoded = layers.Dense(4, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu', name='decode_1')(encoded)
        outputs = layers.Dense(self.input_dim, activation='linear', name='output')(decoded)
        
        self.autoencoder = keras.Model(inputs, outputs, name='tinyml_autoencoder')
        
        # Compile with appropriate optimizer
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return self.autoencoder
    
    def train(self, X_normal, epochs=100, validation_split=0.2, verbose=1):
        """Train the autoencoder on normal data only."""
        print(f"Training TinyML Autoencoder on {len(X_normal)} normal samples...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)
        ]
        
        # Train the model
        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Set anomaly threshold (99th percentile of reconstruction errors)
        train_predictions = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - train_predictions), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 99)
        
        print(f"Anomaly threshold set to: {self.threshold:.6f}")
        
        return history
    
    def predict_anomaly(self, X):
        """Predict anomalies based on reconstruction error."""
        X_scaled = self.scaler.transform(X)
        predictions = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - predictions), axis=1)
        return (reconstruction_errors > self.threshold).astype(int), reconstruction_errors

class TinyMLQuantizer:
    """Handle quantization for TinyML deployment."""
    
    def __init__(self, model, scaler, representative_data):
        self.model = model
        self.scaler = scaler
        self.representative_data = representative_data
        
    def _representative_dataset(self):
        """Generate representative dataset for quantization calibration."""
        # Use scaled representative data
        scaled_data = self.scaler.transform(self.representative_data)
        for sample in scaled_data:
            yield [sample.astype(np.float32).reshape(1, -1)]
    
    def convert_to_tflite_int8(self, output_path):
        """Convert model to INT8 quantized TensorFlow Lite."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self._representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        return tflite_model
    
    def convert_to_tflite_float16(self, output_path):
        """Convert model to Float16 quantized TensorFlow Lite."""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Enable float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        return tflite_model

class TinyMLDeploymentPipeline:
    """Complete deployment pipeline for TinyML bearing fault detection."""
    
    def __init__(self, dataset_path, output_dir="stm32_deployment_complete"):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.feature_extractor = TinyMLFeatureExtractor()
        self.autoencoder = TinyMLAutoencoder()
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_test = None
        
        # Results storage
        self.results = {
            'training_metrics': {},
            'performance_metrics': {},
            'model_sizes': {},
            'deployment_info': {}
        }
        
    def load_nasa_data(self, max_files_per_set=150):
        """Load and process NASA bearing dataset."""
        print("Loading NASA data for TinyML training...")
        
        # Data containers
        all_features = []
        all_labels = []
        
        # Process each test set
        test_sets = ['1st_test', '2nd_test', '3rd_test']
        
        for test_set in test_sets:
            test_path = self.dataset_path / test_set
            if not test_path.exists():
                print(f"Warning: {test_set} directory not found")
                continue
                
            # Get all files in chronological order
            files = sorted([f for f in test_path.iterdir() if f.is_file() and '.' in f.name])
            files = files[:max_files_per_set]
            
            print(f"Processing {len(files)} files from {test_set}")
            
            for i, file_path in enumerate(files):
                if i % 50 == 0:
                    print(f"  Processing file {i+1}/{len(files)}")
                
                try:
                    # Load signal data
                    data = np.loadtxt(file_path, delimiter='\t')
                    if data.shape[1] >= 2:
                        signal = data[:, 1]  # Use bearing channel
                    else:
                        continue
                        
                    # Extract features
                    features = self.feature_extractor.extract_features(signal)
                    all_features.append(features)
                    
                    # Assign labels based on temporal position
                    # Early files are normal, later files show degradation
                    total_files = len(files)
                    normal_threshold = int(0.6 * total_files)  # First 60% are normal
                    
                    if i < normal_threshold:
                        all_labels.append(0)  # Normal
                    else:
                        all_labels.append(1)  # Anomalous
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Handle any NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"Normal samples: {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
        print(f"Anomalous samples: {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
        
        return X, y
    
    def run_complete_pipeline(self):
        """Execute the complete TinyML deployment pipeline."""
        print("🚀 Starting Complete TinyML Deployment Pipeline")
        print("=" * 70)
        
        try:
            # Step 1: Load data
            print("📊 Step 1: Loading and preparing data...")
            X, y = self.load_nasa_data(max_files_per_set=150)
            
            # Split data
            normal_mask = (y == 0)
            anomaly_mask = (y == 1)
            
            self.X_train = X[normal_mask]  # Train only on normal data
            self.X_test = X[anomaly_mask]  # Test on anomalous data
            self.y_test = y[anomaly_mask]
            
            print(f"Training on {len(self.X_train)} normal samples")
            print(f"Testing with {len(self.X_test)} anomalous samples")
            
            # Step 2: Train model
            print("🤖 Step 2: Training TinyML Autoencoder...")
            self.autoencoder.build_model()
            self.autoencoder.autoencoder.summary()
            
            total_params = self.autoencoder.autoencoder.count_params()
            print(f"Total parameters: {total_params}")
            
            history = self.autoencoder.train(self.X_train, epochs=100, verbose=1)
            
            # Step 3: Evaluate performance
            print("📈 Step 3: Evaluating model performance...")
            
            # Test on normal data
            normal_predictions, errors_normal = self.autoencoder.predict_anomaly(self.X_train)
            
            # Test on anomalous data
            anomaly_predictions, errors_anomaly = self.autoencoder.predict_anomaly(self.X_test)
            
            # Calculate metrics
            normal_accuracy = 1 - np.mean(normal_predictions)
            anomaly_detection_rate = np.mean(anomaly_predictions)
            
            # Overall metrics
            all_predictions = np.concatenate([normal_predictions, anomaly_predictions])
            all_true_labels = np.concatenate([np.zeros(len(normal_predictions)), np.ones(len(anomaly_predictions))])
            
            overall_accuracy = accuracy_score(all_true_labels, all_predictions)
            precision = precision_score(all_true_labels, all_predictions, zero_division=0)
            recall = recall_score(all_true_labels, all_predictions, zero_division=0)
            f1 = f1_score(all_true_labels, all_predictions, zero_division=0)
            
            # Store performance metrics
            self.results['performance_metrics'] = {
                'normal_accuracy': float(normal_accuracy),
                'anomaly_detection_rate': float(anomaly_detection_rate),
                'overall_accuracy': float(overall_accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'threshold': float(self.autoencoder.threshold)
            }
            
            print(f"Normal data accuracy: {normal_accuracy*100:.1f}%")
            print(f"Anomaly detection rate: {anomaly_detection_rate*100:.1f}%")
            print(f"Overall accuracy: {overall_accuracy*100:.1f}%")
            print(f"F1-Score: {f1*100:.1f}%")
            
            # Step 4: Apply quantization
            print("⚡ Step 4: Applying heavy quantization...")
            
            quantizer = TinyMLQuantizer(
                model=self.autoencoder.autoencoder,
                scaler=self.autoencoder.scaler,
                representative_data=self.X_train[:500]
            )
            
            # INT8 Quantization
            int8_path = self.output_dir / "bearing_model_int8.tflite"
            int8_model = quantizer.convert_to_tflite_int8(str(int8_path))
            int8_size = len(int8_model) / 1024
            
            # Float16 Quantization
            float16_path = self.output_dir / "bearing_model_float16.tflite"
            float16_model = quantizer.convert_to_tflite_float16(str(float16_path))
            float16_size = len(float16_model) / 1024
            
            print(f"✅ INT8 model: {int8_size:.1f} KB")
            print(f"✅ Float16 model: {float16_size:.1f} KB")
            
            # Step 5: Save all files
            print("💾 Step 5: Saving deployment files...")
            
            # Save scaler
            scaler_path = self.output_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.autoencoder.scaler, f)
                
            # Save threshold
            threshold_path = self.output_dir / "threshold.npy"
            np.save(threshold_path, self.autoencoder.threshold)
            
            # Save Keras model
            keras_path = self.output_dir / "tinyml_autoencoder.keras"
            self.autoencoder.autoencoder.save(keras_path)
            
            # Generate C header
            with open(int8_path, 'rb') as f:
                model_data = f.read()
                
            header_content = f'''/*
 * Auto-generated TensorFlow Lite model data for STM32 deployment
 * Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * Model: NASA Bearing Fault Detection (TinyML Optimized)
 * Size: {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)
 * Quantization: INT8
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

// Model data array
const unsigned char model_data[] = {{
'''
            
            # Add model data as hex bytes
            for i, byte in enumerate(model_data):
                if i % 16 == 0:
                    header_content += "    "
                header_content += f"0x{byte:02x}"
                if i < len(model_data) - 1:
                    header_content += ", "
                if (i + 1) % 16 == 0:
                    header_content += "\\n"
                    
            header_content += f'''
}};

const unsigned int model_data_len = {len(model_data)};

// Model metadata
#define MODEL_INPUT_SIZE 8
#define MODEL_OUTPUT_SIZE 8
#define MODEL_SIZE_BYTES {len(model_data)}
#define MODEL_QUANTIZATION_INT8

// Anomaly detection threshold (Q14 fixed-point)
#define ANOMALY_THRESHOLD_FIXED {int(self.autoencoder.threshold * 16384)}

#endif // MODEL_DATA_H
'''
            
            # Save header file
            header_path = self.output_dir / "model_data.h"
            with open(header_path, 'w') as f:
                f.write(header_content)
            
            # Save results as JSON
            json_path = self.output_dir / "deployment_results.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print("\n🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            print("=" * 70)
            print(f"📁 All files saved in: {self.output_dir.absolute()}")
            print(f"🔧 C header file: {header_path}")
            print(f"⚡ INT8 model: {int8_path} ({int8_size:.1f} KB)")
            print(f"📊 Performance: {overall_accuracy*100:.1f}% accuracy")
            print(f"🎯 Ready for STM32 deployment!")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the complete deployment pipeline."""
    # Configuration
    DATASET_PATH = "D:/errorDetection"
    OUTPUT_DIR = "stm32_deployment_complete"
    
    # Check dataset path
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path not found: {DATASET_PATH}")
        print("Please update DATASET_PATH to your NASA bearing dataset location.")
        return False
    
    # Create deployment pipeline
    pipeline = TinyMLDeploymentPipeline(DATASET_PATH, OUTPUT_DIR)
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎊 Congratulations! Your TinyML model is ready for STM32 deployment.")
        print("\nNext steps:")
        print("1. Review the generated files")
        print("2. Integrate the C code into your STM32 project")
        print("3. Test with real sensor data")
        print("4. Fine-tune threshold if needed")
    else:
        print("\n❌ Deployment pipeline failed. Please check errors above.")
    
    return success

if __name__ == "__main__":
    main()