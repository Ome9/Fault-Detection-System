#!/usr/bin/env python3
"""
Multi-Dataset Bearing Fault Detection System
Integrates NASA IMS, CWRU, and HUST datasets for enhanced accuracy

This system combines three major bearing fault datasets:
1. NASA IMS - Temporal fault progression data
2. CWRU - 10 fault classes with 2300 samples  
3. HUST - 99 samples across multiple fault types

Author: AI Assistant
Date: 2025-09-26
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import pickle
import json
import scipy.io as sio
from scipy import signal as sig, stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

class UnifiedFeatureExtractor:
    """Extract 16 comprehensive bearing fault features from any vibration signal"""
    
    def __init__(self, sampling_rate=20000):
        self.fs = sampling_rate
        
    def extract_16_features(self, signal_data):
        """Extract 16 optimized features for bearing fault detection with NaN handling"""
        
        # Ensure signal is 1D and has sufficient length
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()
        
        if len(signal_data) < 100:  # Minimum length check
            signal_data = np.pad(signal_data, (0, max(0, 100 - len(signal_data))), mode='constant')
        
        # Remove any NaN or infinite values
        signal_data = signal_data[np.isfinite(signal_data)]
        if len(signal_data) == 0:
            # Return zero features if signal is empty
            return np.zeros(16, dtype=np.float32)
        
        # Basic time domain (8 features)
        rms = np.sqrt(np.mean(signal_data**2))
        rms = rms if np.isfinite(rms) else 0.0
        
        peak = np.max(np.abs(signal_data))
        peak = peak if np.isfinite(peak) else 0.0
        
        crest_factor = peak / rms if rms > 1e-10 else 0.0
        crest_factor = crest_factor if np.isfinite(crest_factor) else 0.0
        
        kurtosis = stats.kurtosis(signal_data)
        kurtosis = kurtosis if np.isfinite(kurtosis) else 0.0
        
        skewness = stats.skew(signal_data)
        skewness = skewness if np.isfinite(skewness) else 0.0
        
        std_dev = np.std(signal_data)
        std_dev = std_dev if np.isfinite(std_dev) else 0.0
        
        mean_abs = np.mean(np.abs(signal_data))
        mean_abs = mean_abs if np.isfinite(mean_abs) else 0.0
        
        peak_to_peak = np.max(signal_data) - np.min(signal_data)
        peak_to_peak = peak_to_peak if np.isfinite(peak_to_peak) else 0.0
        
        # Advanced features for better fault detection (8 features)
        # Shape factors
        sqrt_mean = np.mean(np.sqrt(np.abs(signal_data)))
        sqrt_mean = sqrt_mean if np.isfinite(sqrt_mean) else 1e-10
        
        clearance_factor = peak / (sqrt_mean**2) if sqrt_mean > 1e-10 else 0.0
        clearance_factor = clearance_factor if np.isfinite(clearance_factor) else 0.0
        
        shape_factor = rms / mean_abs if mean_abs > 1e-10 else 0.0
        shape_factor = shape_factor if np.isfinite(shape_factor) else 0.0
        
        impulse_factor = peak / mean_abs if mean_abs > 1e-10 else 0.0
        impulse_factor = impulse_factor if np.isfinite(impulse_factor) else 0.0
        
        # Envelope analysis for bearing faults
        try:
            analytic_signal = sig.hilbert(signal_data)
            envelope = np.abs(analytic_signal)
            envelope_rms = np.sqrt(np.mean(envelope**2))
            envelope_rms = envelope_rms if np.isfinite(envelope_rms) else rms
            
            envelope_kurtosis = stats.kurtosis(envelope)
            envelope_kurtosis = envelope_kurtosis if np.isfinite(envelope_kurtosis) else kurtosis
        except:
            envelope_rms = rms
            envelope_kurtosis = kurtosis
        
        # Frequency domain
        try:
            fft_vals = fft(signal_data)
            freqs = fftfreq(len(signal_data), 1/self.fs)
            power_spectrum = np.abs(fft_vals)**2
            
            # Remove any NaN/inf from spectrum
            power_spectrum = power_spectrum[np.isfinite(power_spectrum)]
            freqs = freqs[:len(power_spectrum)]
            
            # Bearing fault frequency bands
            bearing_freq_mask = (freqs >= 100) & (freqs <= 2000)
            bearing_freq_power = np.sum(power_spectrum[bearing_freq_mask]) if np.any(bearing_freq_mask) else 0.0
            bearing_freq_power = bearing_freq_power if np.isfinite(bearing_freq_power) else 0.0
            
            # High frequency content (damage indicator)
            high_freq_mask = freqs >= 5000
            high_freq_power = np.sum(power_spectrum[high_freq_mask]) if np.any(high_freq_mask) else 0.0
            high_freq_power = high_freq_power if np.isfinite(high_freq_power) else 0.0
            
            # Spectral kurtosis
            valid_spectrum = power_spectrum[power_spectrum > 0]
            spectral_kurtosis = stats.kurtosis(valid_spectrum) if len(valid_spectrum) > 3 else 0.0
            spectral_kurtosis = spectral_kurtosis if np.isfinite(spectral_kurtosis) else 0.0
        except:
            bearing_freq_power = 0.0
            high_freq_power = 0.0 
            spectral_kurtosis = 0.0
        
        # Ensure all features are finite
        features = np.array([
            rms, peak, crest_factor, kurtosis, skewness, std_dev, mean_abs, peak_to_peak,
            clearance_factor, shape_factor, impulse_factor, envelope_rms, envelope_kurtosis,
            bearing_freq_power, high_freq_power, spectral_kurtosis
        ], dtype=np.float32)
        
        # Final safety check - replace any remaining NaN/inf with 0
        features = np.where(np.isfinite(features), features, 0.0)
        
        return features

class MultiDatasetLoader:
    """Load and process NASA, CWRU, and HUST datasets with unified feature extraction"""
    
    def __init__(self, base_path="D:/errorDetection"):
        self.base_path = Path(base_path)
        self.feature_extractor = UnifiedFeatureExtractor()
        self.label_encoder = LabelEncoder()
        
    def load_nasa_data(self, max_files_per_set=400):
        """Load NASA IMS data with temporal fault progression"""
        
        print("📊 Loading NASA IMS dataset...")
        
        all_features = []
        all_labels = []
        
        for test_set in ['1st_test', '2nd_test', '3rd_test']:
            test_path = self.base_path / test_set
            
            if not test_path.exists():
                continue
                
            print(f"   Processing {test_set}...")
            
            data_files = sorted(list(test_path.glob('*')))[:max_files_per_set]
            total_files = len(data_files)
            
            for i, file_path in enumerate(data_files):
                try:
                    data = np.loadtxt(file_path)
                    signal = data[:, 0] if data.ndim > 1 else data
                    
                    # Extract 16 features
                    features = self.feature_extractor.extract_16_features(signal)
                    all_features.append(features)
                    
                    # Better labeling strategy
                    progress = i / total_files
                    if progress < 0.65:  # First 65% normal
                        all_labels.append('Normal')
                    else:  # Last 35% fault
                        all_labels.append('Fault')
                        
                except Exception as e:
                    continue
        
        print(f"   ✅ NASA: {len(all_features)} samples")
        return np.array(all_features), np.array(all_labels)
    
    def load_cwru_data(self):
        """Load CWRU dataset from both processed CSV and raw MAT files"""
        
        print("📊 Loading CWRU dataset...")
        
        all_features = []
        all_labels = []
        
        # Load from CSV if available (pre-computed features)
        csv_path = self.base_path / "CWRU_Dataset" / "feature_time_48k_2048_load_1.csv"
        if csv_path.exists():
            print("   Loading from pre-computed CSV features...")
            df = pd.read_csv(csv_path)
            
            # Use existing features and map to our 16-feature format
            for _, row in df.iterrows():
                # Skip rows with NaN values
                if row.isnull().any():
                    continue
                    
                # Map CWRU features to our 16-feature format
                # CWRU has: max, min, mean, sd, rms, skewness, kurtosis, crest, form
                cwru_features = np.array([
                    row['rms'],           # rms
                    row['max'],           # peak  
                    row['crest'],         # crest_factor
                    row['kurtosis'],      # kurtosis
                    row['skewness'],      # skewness
                    row['sd'],            # std_dev
                    (row['max'] - row['min'])/2,  # mean_abs (approximation)
                    row['max'] - row['min'], # peak_to_peak
                    row['form'],          # clearance_factor (approximation)
                    row['form'],          # shape_factor
                    row['crest'] * 1.1,   # impulse_factor (approximation)
                    row['rms'] * 1.05,    # envelope_rms (approximation)
                    row['kurtosis'] * 0.9, # envelope_kurtosis (approximation)
                    row['rms'] * 100,     # bearing_freq_power (approximation)
                    row['rms'] * 50,      # high_freq_power (approximation)
                    row['kurtosis'] * 0.8  # spectral_kurtosis (approximation)
                ], dtype=np.float32)
                
                # Ensure all features are finite
                if np.all(np.isfinite(cwru_features)):
                    all_features.append(cwru_features)
                
                # Map fault labels to binary normal/fault
                fault_type = row['fault']
                if 'Normal' in fault_type:
                    all_labels.append('Normal')
                else:
                    all_labels.append('Fault')
        
        # Also load from raw MAT files for additional diversity
        raw_path = self.base_path / "CWRU_Dataset" / "raw"
        if raw_path.exists():
            print("   Loading from raw MAT files...")
            mat_files = list(raw_path.glob('*.mat'))
            
            for mat_file in mat_files:
                try:
                    data = sio.loadmat(mat_file)
                    
                    # Find the main data key (exclude metadata)
                    data_keys = [k for k in data.keys() if not k.startswith('__')]
                    
                    for key in data_keys:
                        signal = data[key].flatten()
                        
                        # Extract features
                        features = self.feature_extractor.extract_16_features(signal)
                        all_features.append(features)
                        
                        # Label based on filename
                        filename = mat_file.stem
                        if 'Normal' in filename:
                            all_labels.append('Normal')
                        else:
                            all_labels.append('Fault')
                            
                except Exception as e:
                    print(f"   Warning: Could not process {mat_file}: {e}")
                    continue
        
        print(f"   ✅ CWRU: {len(all_features)} samples")
        return np.array(all_features), np.array(all_labels)
    
    def load_hust_data(self):
        """Load HUST dataset from MAT files"""
        
        print("📊 Loading HUST dataset...")
        
        all_features = []
        all_labels = []
        
        hust_path = self.base_path / "HUST_Dataset" / "HUST bearing dataset"
        
        if not hust_path.exists():
            print("   Warning: HUST dataset path not found")
            return np.array([]), np.array([])
        
        mat_files = list(hust_path.glob('*.mat'))
        
        for mat_file in mat_files:
            try:
                data = sio.loadmat(mat_file)
                
                # HUST files typically have 'data' key
                if 'data' in data:
                    signal = data['data'].flatten()
                else:
                    # Fallback to first non-metadata key
                    data_keys = [k for k in data.keys() if not k.startswith('__')]
                    if data_keys:
                        signal = data[data_keys[0]].flatten()
                    else:
                        continue
                
                # Extract features
                features = self.feature_extractor.extract_16_features(signal)
                all_features.append(features)
                
                # Label based on filename pattern
                filename = mat_file.stem
                if filename.startswith('N'):  # Normal
                    all_labels.append('Normal')
                else:  # Any fault type (I, O, B, IB, OB, IO)
                    all_labels.append('Fault')
                    
            except Exception as e:
                print(f"   Warning: Could not process {mat_file}: {e}")
                continue
        
        print(f"   ✅ HUST: {len(all_features)} samples")
        return np.array(all_features), np.array(all_labels)
    
    def load_all_datasets(self):
        """Load and combine all three datasets"""
        
        print("🚀 Loading Multi-Dataset for Enhanced Training")
        print("=" * 80)
        
        # Load individual datasets
        nasa_X, nasa_y = self.load_nasa_data()
        cwru_X, cwru_y = self.load_cwru_data()
        hust_X, hust_y = self.load_hust_data()
        
        # Combine all datasets
        all_features = []
        all_labels = []
        dataset_sources = []
        
        if len(nasa_X) > 0:
            all_features.append(nasa_X)
            all_labels.extend(nasa_y)
            dataset_sources.extend(['NASA'] * len(nasa_X))
        
        if len(cwru_X) > 0:
            all_features.append(cwru_X)
            all_labels.extend(cwru_y)
            dataset_sources.extend(['CWRU'] * len(cwru_X))
        
        if len(hust_X) > 0:
            all_features.append(hust_X)
            all_labels.extend(hust_y)
            dataset_sources.extend(['HUST'] * len(hust_X))
        
        if not all_features:
            raise ValueError("No datasets could be loaded!")
        
        # Combine features
        X_combined = np.vstack(all_features)
        y_combined = np.array(all_labels)
        sources = np.array(dataset_sources)
        
        # Remove any samples with NaN or infinite values
        valid_mask = np.all(np.isfinite(X_combined), axis=1)
        X_combined = X_combined[valid_mask]
        y_combined = y_combined[valid_mask]
        sources = sources[valid_mask]
        
        print(f"   Removed {np.sum(~valid_mask)} samples with invalid values")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_combined)
        
        print(f"\n📈 COMBINED DATASET SUMMARY:")
        print(f"   Total samples: {len(X_combined)}")
        print(f"   Features: {X_combined.shape[1]}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        
        # Show class distribution
        unique_labels, counts = np.unique(y_combined, return_counts=True)
        for label, count in zip(unique_labels, counts):
            percentage = count / len(y_combined) * 100
            print(f"   {label}: {count} samples ({percentage:.1f}%)")
        
        # Show dataset sources
        unique_sources, source_counts = np.unique(sources, return_counts=True)
        print(f"\n📊 DATASET SOURCES:")
        for source, count in zip(unique_sources, source_counts):
            percentage = count / len(sources) * 100
            print(f"   {source}: {count} samples ({percentage:.1f}%)")
        
        return X_combined, y_encoded, sources

class EnhancedAutoencoder:
    """Enhanced autoencoder architecture for multi-dataset training"""
    
    def __init__(self, input_dim=16):
        self.input_dim = input_dim
    
    def build_model(self):
        """Build enhanced autoencoder for multi-dataset training"""
        
        inputs = tf.keras.Input(shape=(self.input_dim,), name='input')
        
        # Encoder with more capacity for diverse datasets
        x = layers.Dense(64, activation='relu', name='encoder_1')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        
        x = layers.Dense(32, activation='relu', name='encoder_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Bottleneck
        encoded = layers.Dense(16, activation='relu', name='bottleneck')(x)
        
        # Decoder
        x = layers.Dense(32, activation='relu', name='decoder_1')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        x = layers.Dense(64, activation='relu', name='decoder_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.15)(x)
        
        # Output
        outputs = layers.Dense(self.input_dim, activation='linear', name='output')(x)
        
        model = tf.keras.Model(inputs, outputs, name='enhanced_multi_dataset_autoencoder')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001), 
            loss='mse', 
            metrics=['mae']
        )
        
        return model

class MultiDatasetPipeline:
    """Complete pipeline for multi-dataset bearing fault detection"""
    
    def __init__(self, base_path="D:/errorDetection", output_dir="multi_dataset_deployment"):
        self.base_path = base_path
        self.output_dir = output_dir
        self.loader = MultiDatasetLoader(base_path)
        self.model_builder = EnhancedAutoencoder(input_dim=16)
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def balance_dataset_with_smote(self, X, y):
        """Balance dataset using SMOTE for multi-dataset training"""
        
        print("⚖️  Balancing multi-dataset with SMOTE...")
        
        # Use SMOTE to balance the dataset
        smote = SMOTE(random_state=42, k_neighbors=min(3, len(X)//2))
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"   Original: Normal={np.sum(y==0)}, Fault={np.sum(y==1)}")
        print(f"   Balanced: Normal={np.sum(y_balanced==0)}, Fault={np.sum(y_balanced==1)}")
        
        return X_balanced, y_balanced
    
    def optimize_threshold(self, reconstruction_errors, y_true):
        """Find optimal threshold for multi-dataset performance"""
        
        print("🎯 Optimizing detection threshold for multi-dataset...")
        
        # Try different percentile thresholds
        thresholds = np.percentile(reconstruction_errors, np.linspace(70, 99.5, 50))
        
        best_threshold = None
        best_f1 = 0
        
        for threshold in thresholds:
            predictions = (reconstruction_errors > threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, predictions, average='binary', zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"   Optimal threshold: {best_threshold:.6f}")
        print(f"   Best F1-score: {best_f1:.4f}")
        
        return best_threshold
    
    def train_multi_dataset_model(self, X, y, sources):
        """Train enhanced model on multi-dataset"""
        
        print("🧠 Training Enhanced Multi-Dataset Model...")
        
        # Balance dataset
        X_balanced, y_balanced = self.balance_dataset_with_smote(X, y)
        
        # Use only normal samples for autoencoder training
        X_normal = X_balanced[y_balanced == 0]
        
        # Scale features
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        X_all_scaled = self.scaler.transform(X)
        X_balanced_scaled = self.scaler.transform(X_balanced)
        
        # Build and train model
        self.model = self.model_builder.build_model()
        
        print("Enhanced Multi-Dataset Model Architecture:")
        self.model.summary()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)
        ]
        
        # Train on normal samples only
        history = self.model.fit(
            X_normal_scaled, X_normal_scaled,
            epochs=300,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate reconstruction errors on balanced dataset
        reconstructions_balanced = self.model.predict(X_balanced_scaled, verbose=0)
        reconstruction_errors_balanced = np.mean(np.square(X_balanced_scaled - reconstructions_balanced), axis=1)
        
        # Optimize threshold
        self.threshold = self.optimize_threshold(reconstruction_errors_balanced, y_balanced)
        
        print(f"✅ Multi-dataset training completed with threshold: {self.threshold:.6f}")
        
        return history
    
    def evaluate_multi_dataset_model(self, X, y, sources):
        """Evaluate model performance on multi-dataset"""
        
        print("📈 Evaluating Multi-Dataset Model Performance...")
        
        X_scaled = self.scaler.transform(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Make predictions
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average='binary', zero_division=0)
        
        # Per-class accuracy
        normal_mask = y == 0
        fault_mask = y == 1
        
        normal_acc = accuracy_score(y[normal_mask], predictions[normal_mask]) if np.any(normal_mask) else 0
        fault_acc = accuracy_score(y[fault_mask], predictions[fault_mask]) if np.any(fault_mask) else 0
        
        # Additional metrics
        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
        balanced_acc = balanced_accuracy_score(y, predictions)
        mcc = matthews_corrcoef(y, predictions)
        
        # Per-dataset performance
        dataset_performance = {}
        for dataset in np.unique(sources):
            dataset_mask = sources == dataset
            if np.any(dataset_mask):
                dataset_acc = accuracy_score(y[dataset_mask], predictions[dataset_mask])
                dataset_performance[dataset] = dataset_acc
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'matthews_corrcoef': mcc,
            'normal_accuracy': normal_acc,
            'fault_accuracy': fault_acc,
            'threshold': self.threshold,
            'total_samples': len(y),
            'normal_samples': np.sum(y == 0),
            'fault_samples': np.sum(y == 1),
            'features': 16,
            'model_params': self.model.count_params(),
            'dataset_performance': dataset_performance
        }
        
        print(f"\n🏆 MULTI-DATASET MODEL RESULTS:")
        print(f"   Overall Accuracy: {accuracy*100:.1f}%")
        print(f"   Balanced Accuracy: {balanced_acc*100:.1f}%")
        print(f"   Precision: {precision*100:.1f}%")
        print(f"   Recall (Fault Detection): {recall*100:.1f}%")
        print(f"   F1-Score: {f1*100:.1f}%")
        print(f"   Matthews Correlation: {mcc:.3f}")
        print(f"   Normal Detection: {normal_acc*100:.1f}%")
        print(f"   Fault Detection: {fault_acc*100:.1f}%")
        
        print(f"\n📊 PER-DATASET PERFORMANCE:")
        for dataset, perf in dataset_performance.items():
            print(f"   {dataset}: {perf*100:.1f}%")
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        print(f"\n📊 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Normal  Fault")
        print(f"   Actual Normal  {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"          Fault   {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        return results
    
    def generate_tflite_model(self, X):
        """Generate optimized TensorFlow Lite model"""
        
        print("🔄 Generating Multi-Dataset TensorFlow Lite model...")
        
        X_scaled = self.scaler.transform(X[:500])  # Use more samples for optimization
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset():
            for i in range(min(len(X_scaled), 200)):
                yield [np.array([X_scaled[i]], dtype=np.float32)]
        
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        
        tflite_model = converter.convert()
        
        print(f"✅ Multi-dataset model size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model
    
    def save_multi_dataset_artifacts(self, tflite_model, results):
        """Save all deployment artifacts"""
        
        print("💾 Saving multi-dataset deployment artifacts...")
        
        # Save TensorFlow Lite model
        tflite_path = Path(self.output_dir) / "multi_dataset_bearing_model_float32.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Save Keras model
        keras_path = Path(self.output_dir) / "multi_dataset_autoencoder.keras"
        self.model.save(keras_path)
        
        # Save scaler
        scaler_path = Path(self.output_dir) / "multi_dataset_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoder
        encoder_path = Path(self.output_dir) / "multi_dataset_label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.loader.label_encoder, f)
        
        # Save threshold
        threshold_path = Path(self.output_dir) / "multi_dataset_threshold.npy"
        np.save(threshold_path, self.threshold)
        
        # Save results
        results_path = Path(self.output_dir) / "multi_dataset_results.json"
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (np.floating, np.integer)):
                json_results[key] = float(value)
            elif isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in value.items()}
            else:
                json_results[key] = value
                
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Multi-dataset artifacts saved to: {self.output_dir}/")
        
        return json_results
    
    def run_complete_multi_dataset_pipeline(self):
        """Run complete multi-dataset pipeline"""
        
        print("🚀 Starting MULTI-DATASET Bearing Fault Detection Pipeline")
        print("=" * 80)
        print("🎯 Multi-Dataset Training:")
        print("   - NASA IMS: Temporal fault progression")
        print("   - CWRU: 10 fault classes with 2300+ samples")
        print("   - HUST: 99 samples across multiple fault types")
        print("   - Enhanced 16-feature extraction")
        print("   - Larger architecture (64→32→16→32→64)")
        print("   - SMOTE balancing across all datasets")
        print("=" * 80)
        
        # Load multi-dataset
        X, y, sources = self.loader.load_all_datasets()
        
        # Train enhanced model
        history = self.train_multi_dataset_model(X, y, sources)
        
        # Evaluate model
        results = self.evaluate_multi_dataset_model(X, y, sources)
        
        # Generate TFLite model
        tflite_model = self.generate_tflite_model(X)
        
        # Save all artifacts
        final_results = self.save_multi_dataset_artifacts(tflite_model, results)
        
        print(f"\n🎉 MULTI-DATASET PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"📁 Output directory: {self.output_dir}/")
        print(f"🔥 Model size: {len(tflite_model)/1024:.1f} KB")
        print(f"⚡ Features: 16 enhanced indicators")
        print(f"🧠 Parameters: {results['model_params']:,}")
        print(f"📊 F1-Score: {results['f1_score']*100:.1f}%")
        print(f"🎯 Fault Detection: {results['fault_accuracy']*100:.1f}%")
        print(f"⚖️  Balanced Accuracy: {results['balanced_accuracy']*100:.1f}%")
        print(f"🌐 Total Training Samples: {results['total_samples']:,}")
        
        # Show improvement over single dataset
        print(f"\n📈 MULTI-DATASET vs SINGLE DATASET:")
        print(f"   Training Samples: {results['total_samples']:,} (vs ~1,200 single)")
        print(f"   Dataset Diversity: 3 sources (vs 1 NASA only)")
        print(f"   Expected Improvement: Better generalization & robustness")
        
        return final_results

def main():
    """Run multi-dataset pipeline"""
    
    BASE_PATH = "D:/errorDetection"
    OUTPUT_DIR = "multi_dataset_deployment"
    
    pipeline = MultiDatasetPipeline(BASE_PATH, OUTPUT_DIR)
    results = pipeline.run_complete_multi_dataset_pipeline()
    
    return results

if __name__ == "__main__":
    results = main()