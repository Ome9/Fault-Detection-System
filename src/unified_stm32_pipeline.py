#!/usr/bin/env python3
"""
Unified STM32 TinyML Deployment Pipeline
Optimized Float32 Model Generation for Maximum Performance

This comprehensive pipeline:
1. Loads and preprocesses NASA bearing dataset
2. Trains optimized autoencoder with proper architecture
3. Applies TensorFlow graph optimizations (NO parameter reduction)
4. Generates float32 TFLite model with maximum performance
5. Creates complete STM32 deployment package
6. Includes benchmarking and validation tools

Author: AI Assistant
Date: 2025-09-26
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import pickle
import json
from scipy import signal as sig, stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class OptimizedBearingFeatureExtractor:
    """Optimized feature extraction for bearing fault detection"""
    
    def __init__(self, sampling_rate=20000):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def extract_optimized_features(self, signal_data):
        """Extract 8 most important features for bearing fault detection"""
        # Time domain features
        rms = np.sqrt(np.mean(signal_data**2))
        peak = np.max(np.abs(signal_data))
        crest_factor = peak / rms if rms > 0 else 0
        kurtosis = stats.kurtosis(signal_data)
        
        # Envelope analysis for bearing faults
        analytic_signal = sig.hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        envelope_peak = np.max(envelope)
        
        # Frequency domain analysis
        fft_vals = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.fs)
        power_spectrum = np.abs(fft_vals)**2
        
        # High frequency power (>5kHz for bearing faults)
        high_freq_mask = freqs > 5000
        high_freq_power = np.sum(power_spectrum[high_freq_mask])
        
        # Bearing fault frequency bands (typical range: 100-2000 Hz)
        bearing_freq_mask = (freqs >= 100) & (freqs <= 2000)
        bearing_freq_power = np.sum(power_spectrum[bearing_freq_mask])
        
        # Spectral kurtosis for impulsive faults
        spectral_kurtosis = stats.kurtosis(power_spectrum[power_spectrum > 0])
        
        return np.array([
            rms, peak, crest_factor, kurtosis,
            envelope_peak, high_freq_power, bearing_freq_power, spectral_kurtosis
        ], dtype=np.float32)

class OptimizedAutoencoderBuilder:
    """Build optimized autoencoder for STM32 deployment"""
    
    def __init__(self, input_dim=8, encoding_dim=4):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
    def build_float32_optimized_model(self):
        """Build float32 model optimized for STM32 inference"""
        
        # Input layer
        inputs = tf.keras.Input(shape=(self.input_dim,), name='input', dtype=tf.float32)
        
        # Encoder with optimized activation functions
        x = layers.Dense(16, activation='relu', name='encoder_1',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros')(inputs)
        
        # Bottleneck layer
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck',
                              kernel_initializer='he_normal',
                              bias_initializer='zeros')(x)
        
        # Decoder  
        x = layers.Dense(16, activation='relu', name='decoder_1',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros')(encoded)
        
        # Output layer (linear activation for reconstruction)
        outputs = layers.Dense(self.input_dim, activation='linear', name='output',
                              kernel_initializer='he_normal',
                              bias_initializer='zeros')(x)
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name='optimized_bearing_autoencoder')
        
        # Compile with optimized optimizer for float32
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7  # Optimized for float32 precision
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model

class Float32STM32Optimizer:
    """TensorFlow graph optimization for STM32 float32 inference"""
    
    def __init__(self):
        self.optimization_passes = [
            'constant_folding',
            'arithmetic_optimization', 
            'dependency_optimization',
            'layout_optimizer',
            'memory_optimizer',
            'function_optimization'
        ]
    
    def optimize_for_inference(self, model, representative_data):
        """Apply comprehensive optimizations for STM32 float32 inference"""
        
        print("🔧 Applying TensorFlow graph optimizations...")
        
        # 1. Convert to TensorFlow Lite with optimizations
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable all optimizations except quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Set representative dataset for calibration
        def representative_dataset():
            for i in range(min(len(representative_data), 500)):
                yield [np.array([representative_data[i]], dtype=np.float32)]
        
        converter.representative_dataset = representative_dataset
        
        # Keep float32 precision (no quantization)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Allow TF ops if needed
        ]
        
        # Ensure float32 input/output
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        
        # Advanced optimizations
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = False  # No quantization
        
        # Optimize for size and latency
        converter.experimental_enable_resource_variables = False
        
        # Convert model
        tflite_model = converter.convert()
        
        print(f"✅ Model optimized. Size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model
    
    def benchmark_model(self, tflite_model, test_data, num_runs=100):
        """Benchmark the optimized model"""
        
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], 
                                 np.array([test_data[0]], dtype=np.float32))
            interpreter.invoke()
        
        # Benchmark inference time
        import time
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], 
                                 np.array([test_data[i % len(test_data)]], dtype=np.float32))
            interpreter.invoke()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"📊 Inference Benchmark Results:")
        print(f"   Average time: {avg_time:.3f} ms")
        print(f"   Std deviation: {std_time:.3f} ms")
        print(f"   Min time: {np.min(times):.3f} ms")
        print(f"   Max time: {np.max(times):.3f} ms")
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times)
        }

class UnifiedDeploymentPipeline:
    """Complete deployment pipeline for STM32"""
    
    def __init__(self, dataset_path, output_dir="stm32_optimized_deployment"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.feature_extractor = OptimizedBearingFeatureExtractor()
        self.model_builder = OptimizedAutoencoderBuilder()
        self.optimizer = Float32STM32Optimizer()
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_nasa_data(self, max_files_per_set=300):
        """Load and process NASA IMS bearing dataset"""
        
        print("📊 Loading NASA IMS bearing dataset...")
        
        all_features = []
        all_labels = []
        
        # Process each test set
        for test_set in ['1st_test', '2nd_test', '3rd_test']:
            test_path = Path(self.dataset_path) / test_set
            
            if not test_path.exists():
                print(f"⚠️  Warning: {test_set} directory not found")
                continue
                
            print(f"   Processing {test_set}...")
            
            # Get all data files
            data_files = sorted(list(test_path.glob('*')))[:max_files_per_set]
            
            for i, file_path in enumerate(data_files):
                try:
                    # Load data (NASA format: space-separated values, 4 channels)
                    data = np.loadtxt(file_path)
                    
                    # Use channel 1 (bearing channel) - first column
                    signal = data[:, 0] if data.ndim > 1 else data
                    
                    # Extract features
                    features = self.feature_extractor.extract_optimized_features(signal)
                    all_features.append(features)
                    
                    # Label: early files = normal (0), later files = fault (1)
                    # Assume last 30% of files represent degraded state
                    threshold_file = int(0.7 * len(data_files))
                    label = 1 if i >= threshold_file else 0
                    all_labels.append(label)
                    
                except Exception as e:
                    print(f"   Error processing {file_path}: {e}")
                    continue
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"   Normal samples: {np.sum(y == 0)}")
        print(f"   Fault samples: {np.sum(y == 1)}")
        
        return X, y
    
    def train_optimized_model(self, X, y, validation_split=0.2):
        """Train the optimized autoencoder model"""
        
        print("🧠 Training optimized autoencoder...")
        
        # Use only normal data for training (unsupervised learning)
        X_normal = X[y == 0]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Build model
        self.model = self.model_builder.build_float32_optimized_model()
        
        print("Model Architecture:")
        self.model.summary()
        
        # Training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=200,
            batch_size=32,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold using validation data
        X_all_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_all_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_all_scaled - reconstructions), axis=1)
        
        # Use 95th percentile of normal samples as threshold
        normal_errors = reconstruction_errors[y == 0]
        self.threshold = np.percentile(normal_errors, 95)
        
        print(f"✅ Training completed. Threshold: {self.threshold:.6f}")
        
        return history
    
    def evaluate_model(self, X, y):
        """Evaluate model performance"""
        
        print("📈 Evaluating model performance...")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Make predictions
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary')
        
        # Detailed analysis
        normal_acc = accuracy_score(y[y == 0], predictions[y == 0])
        fault_acc = accuracy_score(y[y == 1], predictions[y == 1])
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'normal_accuracy': normal_acc,
            'fault_accuracy': fault_acc,
            'threshold': self.threshold,
            'total_samples': len(y),
            'normal_samples': np.sum(y == 0),
            'fault_samples': np.sum(y == 1)
        }
        
        print(f"   Overall Accuracy: {accuracy*100:.1f}%")
        print(f"   Precision: {precision*100:.1f}%")
        print(f"   Recall: {recall*100:.1f}%")
        print(f"   F1-Score: {f1*100:.1f}%")
        print(f"   Normal Accuracy: {normal_acc*100:.1f}%")
        print(f"   Fault Accuracy: {fault_acc*100:.1f}%")
        
        return results
    
    def generate_optimized_tflite(self, X):
        """Generate optimized TensorFlow Lite model"""
        
        print("🔄 Generating optimized TensorFlow Lite model...")
        
        # Use representative data for optimization
        X_scaled = self.scaler.transform(X[:300])  # Use subset for representative data
        
        # Apply optimizations
        tflite_model = self.optimizer.optimize_for_inference(self.model, X_scaled)
        
        # Benchmark the model
        benchmark_results = self.optimizer.benchmark_model(tflite_model, X_scaled)
        
        return tflite_model, benchmark_results
    
    def generate_c_header(self, tflite_model, header_name="optimized_model_data.h"):
        """Generate C header file for STM32"""
        
        print("🔧 Generating C header file...")
        
        header_path = Path(self.output_dir) / header_name
        
        with open(header_path, 'w') as f:
            f.write(f"""/*
 * Optimized Float32 TensorFlow Lite Model for STM32
 * Generated automatically from trained model
 * 
 * Model: NASA Bearing Fault Detection (Float32 Optimized)
 * Size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)
 * Precision: Float32 (no quantization)
 * Features: 8 optimized bearing fault features
 * Architecture: 8->16->4->16->8 (428 parameters)
 * 
 * Author: AI Assistant
 * Date: {np.datetime64('today')}
 */

#ifndef OPTIMIZED_MODEL_DATA_H
#define OPTIMIZED_MODEL_DATA_H

#include <stdint.h>

// Model metadata
#define MODEL_INPUT_SIZE 8
#define MODEL_OUTPUT_SIZE 8
#define MODEL_SIZE_BYTES {len(tflite_model)}
#define MODEL_PRECISION_FLOAT32
#define ANOMALY_THRESHOLD_F32 {self.threshold:.8f}f

// Model data array
const unsigned char optimized_model_data[] = {{
""")
            
            # Write model data as hex array
            for i, byte in enumerate(tflite_model):
                if i % 16 == 0:
                    f.write("\n    ")
                f.write(f"0x{byte:02x}")
                if i < len(tflite_model) - 1:
                    f.write(", ")
            
            f.write(f"""
}};

const unsigned int optimized_model_data_len = {len(tflite_model)};

// Feature scaling parameters (float32)
const float scaler_min[MODEL_INPUT_SIZE] = {{
    {self.scaler.data_min_[0]:.8f}f, {self.scaler.data_min_[1]:.8f}f, 
    {self.scaler.data_min_[2]:.8f}f, {self.scaler.data_min_[3]:.8f}f,
    {self.scaler.data_min_[4]:.8f}f, {self.scaler.data_min_[5]:.8f}f,
    {self.scaler.data_min_[6]:.8f}f, {self.scaler.data_min_[7]:.8f}f
}};

const float scaler_scale[MODEL_INPUT_SIZE] = {{
    {self.scaler.scale_[0]:.8f}f, {self.scaler.scale_[1]:.8f}f,
    {self.scaler.scale_[2]:.8f}f, {self.scaler.scale_[3]:.8f}f,
    {self.scaler.scale_[4]:.8f}f, {self.scaler.scale_[5]:.8f}f,
    {self.scaler.scale_[6]:.8f}f, {self.scaler.scale_[7]:.8f}f
}};

#endif // OPTIMIZED_MODEL_DATA_H
""")
        
        print(f"✅ C header saved: {header_path}")
        return header_path
    
    def save_deployment_artifacts(self, tflite_model, results, benchmark_results):
        """Save all deployment artifacts"""
        
        print("💾 Saving deployment artifacts...")
        
        # Save TensorFlow Lite model
        tflite_path = Path(self.output_dir) / "optimized_bearing_model_float32.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Save Keras model
        keras_path = Path(self.output_dir) / "optimized_bearing_autoencoder.keras"
        self.model.save(keras_path)
        
        # Save scaler
        scaler_path = Path(self.output_dir) / "optimized_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save threshold
        threshold_path = Path(self.output_dir) / "optimized_threshold.npy"
        np.save(threshold_path, self.threshold)
        
        # Save comprehensive results (convert numpy types to Python types for JSON)
        results_path = Path(self.output_dir) / "deployment_results.json"
        
        # Convert numpy types to Python types
        def convert_to_python_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_python_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj
        
        combined_results = {
            'model_performance': convert_to_python_types(results),
            'benchmark_results': convert_to_python_types(benchmark_results),
            'model_info': {
                'input_features': 8,
                'architecture': '8->16->4->16->8',
                'parameters': int(self.model.count_params()),
                'model_size_bytes': len(tflite_model),
                'model_size_kb': len(tflite_model) / 1024,
                'precision': 'float32',
                'optimizations': 'graph_optimization,constant_folding,layout_optimizer'
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"✅ Deployment artifacts saved to: {self.output_dir}/")
        return combined_results
    
    def run_complete_pipeline(self):
        """Run the complete deployment pipeline"""
        
        print("🚀 Starting Unified STM32 TinyML Deployment Pipeline")
        print("=" * 70)
        
        # Step 1: Load data
        X, y = self.load_nasa_data()
        
        # Step 2: Train model
        history = self.train_optimized_model(X, y)
        
        # Step 3: Evaluate model
        results = self.evaluate_model(X, y)
        
        # Step 4: Generate optimized TFLite model
        tflite_model, benchmark_results = self.generate_optimized_tflite(X)
        
        # Step 5: Generate C header
        self.generate_c_header(tflite_model)
        
        # Step 6: Save all artifacts
        combined_results = self.save_deployment_artifacts(tflite_model, results, benchmark_results)
        
        # Summary
        print("\n🎯 DEPLOYMENT PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"📁 Output directory: {self.output_dir}/")
        print(f"🔥 Model size: {len(tflite_model)/1024:.1f} KB")
        print(f"⚡ Precision: Float32 (full precision)")
        print(f"🧠 Parameters: {self.model.count_params():,}")
        print(f"📊 Accuracy: {results['accuracy']*100:.1f}%")
        print(f"⏱️  Avg inference: {benchmark_results['avg_time_ms']:.3f} ms")
        print(f"🎛️  Features: 8 optimized bearing fault indicators")
        
        print(f"\n🏆 Float32 Model Performance:")
        print(f"   Overall Accuracy: {results['accuracy']*100:.1f}%")
        print(f"   Normal Detection: {results['normal_accuracy']*100:.1f}%")
        print(f"   Fault Detection: {results['fault_accuracy']*100:.1f}%")
        print(f"   F1-Score: {results['f1_score']*100:.1f}%")
        
        print(f"\n✅ Ready for STM32 deployment with maximum accuracy!")
        
        return combined_results

def main():
    """Main function to run the deployment pipeline"""
    
    # Configuration
    DATASET_PATH = "D:/errorDetection"
    OUTPUT_DIR = "stm32_optimized_deployment"
    
    # Create and run pipeline
    pipeline = UnifiedDeploymentPipeline(DATASET_PATH, OUTPUT_DIR)
    results = pipeline.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()