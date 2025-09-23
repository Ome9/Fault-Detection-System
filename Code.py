#!/usr/bin/env python3
"""
NASA IMS Bearing Dataset Autoencoder-Based Fault Detection System
Optimized for STM32 F446RE Deployment

Dataset Info:
- NASA IMS (Intelligent Maintenance Systems) Bearing Dataset
- 3 Test sets with run-to-failure experiments
- 20kHz sampling rate, 20,480 points per file (1-second snapshots)
- Multiple bearing failures: inner race, outer race, roller element

Author: AI Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob
from scipy import signal as sig, stats
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
import pickle
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NASABearingFeatureExtractor:
    """
    Advanced feature extraction optimized for NASA IMS bearing dataset
    Designed for STM32 F446RE deployment.
    """
    
    def __init__(self, sampling_rate=20000, target_features=16):
        self.fs = sampling_rate
        self.target_features = target_features
        # Define the selected features list as a single source of truth for the project.
        # In your NASABearingFeatureExtractor class __init__ method:

        self.selected_features_list = [
            'rms',                          # Overall vibration level (a better 'energy' measure)
            'peak',                         # Impact detection
            'crest_factor',                 # Impulsiveness
            'kurtosis',                     # Impulsiveness measure
            'spectral_centroid',            # Frequency content shift
            'spectral_kurtosis',            # Spectral impulsiveness
            'high_freq_1_power',            # Inner race fault band
            'high_freq_2_power',            # Outer race fault band
            'bearing_1x_power',             # Shaft-related anomalies
            'bearing_2x_power',             # 2x shaft anomalies
            'envelope_spectral_kurtosis',   # *** Powerful fault indicator ***
            'envelope_peak',                # *** Peak impact in envelope ***
            'std',                          # Variability
            'skewness',                     # ADDED BACK: Asymmetry of the signal
            'impulse_factor',               # Impulse strength
            'wavelet_energy_ratio'          # High/low frequency energy ratio
            # 'energy' has been REMOVED
        ]

    def extract_envelope_features(self, signal):
        features = {}
        # Apply the Hilbert transform to get the analytic signal
        analytic_signal = hilbert(signal)
        # The envelope is the magnitude of the analytic signal
        envelope = np.abs(analytic_signal)
        
        # Now, treat the envelope as a new signal and get its FFT
        envelope_fft_vals = fft(envelope - np.mean(envelope)) # Subtract mean
        envelope_fft_magnitude = np.abs(envelope_fft_vals[:len(envelope_fft_vals)//2])
        
        # These new features will be very sensitive to periodic impacts
        features['envelope_peak'] = np.max(envelope_fft_magnitude)
        features['envelope_spectral_kurtosis'] = stats.kurtosis(envelope_fft_magnitude)
        return features

    def extract_time_domain_features(self, signal):
        """Extract time-domain statistical features from a raw signal."""
        features = {}
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['peak'] = np.max(np.abs(signal))
        features['peak_to_peak'] = np.ptp(signal)
        mean_abs = np.mean(np.abs(signal))
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
        features['clearance_factor'] = features['peak'] / (mean_abs**2) if mean_abs > 0 else 0
        features['impulse_factor'] = features['peak'] / mean_abs if mean_abs > 0 else 0
        features['shape_factor'] = features['rms'] / mean_abs if mean_abs > 0 else 0
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)
        features['energy'] = np.sum(signal**2)
        features['log_energy'] = np.log(features['energy']) if features['energy'] > 0 else 0
        return features
    
    def extract_frequency_domain_features(self, signal):
        """Extract frequency-domain features optimized for bearing faults."""
        features = {}
        fft_vals = fft(signal)
        fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = fftfreq(len(signal), 1/self.fs)[:len(fft_vals)//2]
        total_power = np.sum(fft_magnitude**2)
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0
        features['spectral_spread'] = np.sqrt(np.sum((freqs - features['spectral_centroid'])**2 * fft_magnitude) / np.sum(fft_magnitude)) if np.sum(fft_magnitude) > 0 else 0
        features['spectral_skewness'] = stats.skew(fft_magnitude)
        features['spectral_kurtosis'] = stats.kurtosis(fft_magnitude)
        bearing_bands = {'low_freq': (5, 100), 'bearing_1x': (20, 50), 'bearing_2x': (60, 90), 'high_freq_1': (100, 500), 'high_freq_2': (500, 1500), 'very_high': (1500, 5000)}
        for band_name, (low, high) in bearing_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_power = np.sum(fft_magnitude[band_mask]**2)
                features[f'{band_name}_power'] = band_power / total_power if total_power > 0 else 0
                features[f'{band_name}_peak'] = np.max(fft_magnitude[band_mask])
            else:
                features[f'{band_name}_power'] = 0
                features[f'{band_name}_peak'] = 0
        dominant_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_idx]
        features['dominant_magnitude'] = fft_magnitude[dominant_idx]
        return features
    
    def extract_wavelet_features(self, signal):
        """Extract wavelet-based features (simplified for MCU) using filtering."""
        features = {}
        sos_low = sig.butter(4, 100, btype='low', fs=self.fs, output='sos')
        low_freq = sig.sosfilt(sos_low, signal)
        features['wavelet_low_energy'] = np.sum(low_freq**2)
        sos_high = sig.butter(4, 100, btype='high', fs=self.fs, output='sos')
        high_freq = sig.sosfilt(sos_high, signal)
        features['wavelet_high_energy'] = np.sum(high_freq**2)
        total_energy = features['wavelet_low_energy'] + features['wavelet_high_energy']
        features['wavelet_energy_ratio'] = features['wavelet_high_energy'] / total_energy if total_energy > 0 else 0
        return features
    
    def extract_optimized_features(self, signal):
        """
        Extracts and combines a compact set of features for deployment.
        
        This function calls all individual feature extraction methods (time, frequency,
        wavelet, and envelope), combines their results, and then selects the final
        set of features defined in the class's `selected_features_list`.
        """
        # Step 1: Call all individual feature extraction methods
        time_features = self.extract_time_domain_features(signal)
        freq_features = self.extract_frequency_domain_features(signal)
        wavelet_features = self.extract_wavelet_features(signal)
        envelope_features = self.extract_envelope_features(signal)

        # Step 2: Combine all feature dictionaries into a single master dictionary
        all_features = {
            **time_features, 
            **freq_features, 
            **wavelet_features, 
            **envelope_features
        }
        
        # Step 3: Use the predefined list from __init__ to ensure order and selection
        selected_features = self.selected_features_list
        
        # Optional: Trim the list if it exceeds the target number of features
        if len(selected_features) > self.target_features:
            selected_features = selected_features[:self.target_features]
        
        # Step 4: Build the final feature vector in the correct order
        # Using .get(name, 0.0) ensures a default value if a feature is missing
        feature_vector = [all_features.get(name, 0.0) for name in selected_features]
        
        return np.array(feature_vector), selected_features

class NASABearingDataLoader:
    """Loads and processes the NASA IMS bearing dataset, handling all test sets."""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.feature_extractor = NASABearingFeatureExtractor()
        
    def parse_filename(self, filename):
        """Parses a NASA IMS filename to extract its timestamp."""
        try:
            parts = filename.replace('.txt', '').split('.')
            if len(parts) >= 6:
                year, month, day, hour, minute, second = parts[:6]
                timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                return pd.to_datetime(timestamp)
            return None
        except Exception:
            return None
    
    def load_single_file(self, filepath):
        """Loads a single data file, handling multiple channels."""
        try:
            data = np.loadtxt(filepath)
            if data.ndim == 1:
                return [data]
            else:
                return [data[:, i] for i in range(data.shape[1])]
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def determine_health_condition(self, filepath, test_set):
        """Determines bearing health based on filename and run-to-failure progression."""
        filename = filepath.name
        timestamp = self.parse_filename(filename)
        if timestamp is None: return 'unknown'

        if test_set == 1:
            start_time, end_time = pd.to_datetime('2003-10-22'), pd.to_datetime('2003-11-25')
            degraded_threshold, early_threshold = 0.85, 0.70
        elif test_set == 2:
            start_time, end_time = pd.to_datetime('2004-02-12'), pd.to_datetime('2004-02-19')
            degraded_threshold, early_threshold = 0.80, 0.65
        elif test_set == 3:
            start_time, end_time = pd.to_datetime('2004-03-04'), pd.to_datetime('2004-04-04')
            degraded_threshold, early_threshold = 0.85, 0.70
        else:
            return 'unknown'
        
        total_duration = (end_time - start_time).total_seconds()
        current_progress = (timestamp - start_time).total_seconds() / total_duration if total_duration > 0 else 0
        
        if current_progress > degraded_threshold: return 'degraded'
        if current_progress > early_threshold: return 'early_degraded'
        return 'normal'
    
    def load_test_set(self, test_set_num, max_files=None):
        """Loads all data from a specific test set directory."""
        print(f"Loading NASA IMS Test Set {test_set_num}...")
        test_dirs = list(self.dataset_path.glob(f"*{test_set_num}*"))
        if not test_dirs:
            raise FileNotFoundError(f"Test set {test_set_num} directory not found in {self.dataset_path}")
        
        test_dir = test_dirs[0]
        test_files = sorted(list(test_dir.glob("2*")))
        print(f"Found {len(test_files)} files in test set {test_set_num}")
        if max_files: test_files = test_files[:max_files]
            
        dataset = []
        feature_names = self.feature_extractor.selected_features_list
        
        for i, filepath in enumerate(test_files):
            if i % 100 == 0: print(f"Processing file {i}/{len(test_files)}: {filepath.name}")
            
            channel_data = self.load_single_file(filepath)
            if channel_data is None: continue
            
            condition = self.determine_health_condition(filepath, test_set_num)
            for channel_idx, signal_data in enumerate(channel_data):
                if len(signal_data) == 20480:
                    features, _ = self.feature_extractor.extract_optimized_features(signal_data)
                    dataset.append({
                        'features': features, 'condition': condition, 'test_set': test_set_num,
                        'channel': channel_idx + 1, 'filename': filepath.name,
                        'bearing_num': self.get_bearing_number(channel_idx, test_set_num)
                    })
        
        print(f"Loaded {len(dataset)} samples from test set {test_set_num}")
        return dataset, feature_names
    
    def get_bearing_number(self, channel_idx, test_set):
        """Maps a channel index to its corresponding bearing number."""
        return (channel_idx // 2) + 1 if test_set == 1 else channel_idx + 1
    
    def load_all_data(self, test_sets=[1, 2, 3], max_files_per_set=None):
        """Loads and combines data from multiple test sets."""
        all_data, feature_names = [], None
        for test_set in test_sets:
            try:
                data, names = self.load_test_set(test_set, max_files_per_set)
                all_data.extend(data)
                if feature_names is None: feature_names = names
            except Exception as e:
                print(f"Error loading test set {test_set}: {e}")
        return all_data, feature_names

class BearingAutoencoder:
    """A Deep Autoencoder model optimized for bearing fault detection."""
    
    def __init__(self, input_dim=16, encoding_dim=8):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder, self.encoder = None, None
        self.scaler = MinMaxScaler()
        self.threshold = None
        
    def build_autoencoder(self):
        """Builds the Keras model architecture with regularization."""
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dropout(0.1)(encoded)
        encoded = Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        decoded = Dense(16, activation='relu')(encoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dropout(0.1)(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = BatchNormalization()(decoded)
        output_layer = Dense(self.input_dim, activation='linear')(decoded)
        
        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, encoded)
        self.autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return self.autoencoder
    
    def train(self, X_normal, epochs=100, batch_size=32):
        """Trains the autoencoder on healthy bearing data."""
        print(f"Training Autoencoder on {len(X_normal)} Normal Bearing Samples...")
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        if self.autoencoder is None: self.build_autoencoder()
        
        callbacks = [EarlyStopping(patience=15, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)]
        
        self.autoencoder.fit(X_normal_scaled, X_normal_scaled, epochs=epochs, batch_size=batch_size,
                             validation_split=0.1, callbacks=callbacks, verbose=1)
        
        reconstructed = self.autoencoder.predict(X_normal_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_normal_scaled - reconstructed), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95) # Use 99.5 percentile for better robustness
        print(f"Reconstruction threshold set to: {self.threshold:.6f}")
    
    def predict_anomaly(self, X):
        """Predicts anomalies for a given feature set using the reconstruction error."""
        if not all([self.autoencoder, self.scaler, self.threshold is not None]):
            raise ValueError("Model is not trained or scaler/threshold is missing.")
        
        X_scaled = self.scaler.transform(X)
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)
        anomalies = reconstruction_errors > self.threshold
        return reconstruction_errors, anomalies

class NASABearingFaultDetector:
    """Orchestrates the entire fault detection system from data loading to deployment code generation."""
    
    def __init__(self, dataset_path, test_sets=[1, 2, 3]):
        self.dataset_path = dataset_path
        self.test_sets = test_sets
        self.data_loader = NASABearingDataLoader(dataset_path)
        self.autoencoder = BearingAutoencoder()
        self.feature_names = None
        self.data = {}
        
    def load_and_prepare_data(self, max_files_per_set=None):
        """Loads, preprocesses, and summarizes the dataset."""
        print("Loading and Preparing NASA IMS Bearing Dataset...")
        raw_data, feature_names = self.data_loader.load_all_data(self.test_sets, max_files_per_set)
        self.feature_names = feature_names
        
        features = np.array([sample['features'] for sample in raw_data])
        conditions = np.array([sample['condition'] for sample in raw_data])
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        self.data = {'features': features, 'conditions': conditions, 'raw_data': raw_data}
        unique, counts = np.unique(conditions, return_counts=True)
        print("\nDataset Statistics:")
        for cond, count in zip(unique, counts):
            print(f"  {cond}: {count} samples ({count/len(conditions)*100:.1f}%)")
    
    def train_model(self):
        """Trains the autoencoder model on the prepared data."""
        if not self.data: self.load_and_prepare_data()
        
        normal_mask = self.data['conditions'] == 'normal'
        X_normal = self.data['features'][normal_mask]
        
        self.autoencoder.input_dim = X_normal.shape[1]
        self.autoencoder.train(X_normal, epochs=100)
    
    def evaluate_model(self):
        """Evaluates the trained model's performance and plots results."""
        features, conditions = self.data['features'], self.data['conditions']
        reconstruction_errors, predictions = self.autoencoder.predict_anomaly(features)
        y_true = (conditions != 'normal').astype(int)
        
        print("\n=== Model Evaluation Results ===")
        print(classification_report(y_true, predictions, target_names=['Normal', 'Anomaly']))
        print(f"Anomaly Threshold: {self.autoencoder.threshold:.6f}")
        
        self.plot_evaluation_results(reconstruction_errors, conditions, predictions, y_true)
    
    def plot_evaluation_results(self, reconstruction_errors, conditions, predictions, y_true):
        """Visualizes the model's performance with comprehensive plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Evaluation', fontsize=16)

        # Plot 1: Reconstruction Error Distribution
        sns.histplot(data=pd.DataFrame({'error': reconstruction_errors, 'condition': conditions}),
                     x='error', hue='condition', multiple='stack', ax=axes[0, 0], bins=50,
                     palette={'normal': 'green', 'early_degraded': 'orange', 'degraded': 'red', 'unknown': 'gray'})
        axes[0, 0].axvline(self.autoencoder.threshold, color='black', linestyle='--', label=f'Threshold')
        axes[0, 0].set_title('Reconstruction Error Distribution by Condition')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        
        # Plot 2: Confusion Matrix
        cm = confusion_matrix(y_true, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], 
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted Label')
        axes[0, 1].set_ylabel('True Label')
        
        # Plot 3: Time Series of Reconstruction Errors
        plot_df = pd.DataFrame({'error': reconstruction_errors, 'condition': conditions}).head(2000)
        sns.scatterplot(data=plot_df, x=plot_df.index, y='error', hue='condition', ax=axes[1, 0], s=15,
                        palette={'normal': 'green', 'early_degraded': 'orange', 'degraded': 'red', 'unknown': 'gray'})
        axes[1, 0].axhline(self.autoencoder.threshold, color='black', linestyle='--')
        axes[1, 0].set_title('Reconstruction Error Timeline (First 2000 Samples)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Reconstruction Error')
        
        # Hide the last unused subplot
        axes[1, 1].set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    def generate_stm32_code(self, output_path="stm32_nasa_bearing_detection.c"):
        """Generates a complete, self-contained C code file for STM32 deployment."""
        feature_comments = "\n".join([f"// features[{i}]: {name}" for i, name in enumerate(self.feature_names)])
        scaler_mean = self.autoencoder.scaler.mean_
        scaler_scale = self.autoencoder.scaler.scale_
        threshold = self.autoencoder.threshold
        
        c_code = f"""/*
 * NASA IMS Bearing Fault Detection System for STM32 F446RE
 * Generated on: {pd.Timestamp.now()}
 *
 * This is a self-contained demonstration file. For a real deployment,
 * it is highly recommended to convert the trained model to TensorFlow Lite for Microcontrollers
 * and use the optimized TFLM interpreter for inference. This manual implementation is for
 * educational and prototyping purposes.
 */
#include "stm32f4xx_hal.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// --- Configuration ---
#define SAMPLE_RATE_HZ           1000    // Downsampled for MCU
#define WINDOW_SIZE              1024    // Analysis window size (e.g., ~1 second)
#define NUM_FEATURES             {len(self.feature_names)}
#define ENCODING_DIM             {self.autoencoder.encoding_dim}
#define RECONSTRUCTION_THRESHOLD {threshold:.8f}f

// --- Feature Index Mapping (from Python script) ---
{feature_comments}
// ----------------------------------------------------

// --- Preprocessing Parameters (Scaler) ---
const float scaler_mean[NUM_FEATURES] = {{{', '.join([f'{x:.8f}f' for x in scaler_mean])}}};
const float scaler_scale[NUM_FEATURES] = {{{', '.join([f'{x:.8f}f' for x in scaler_scale])}}};

// --- Model Parameters (Placeholders) ---
// NOTE: These weights are placeholders. A real deployment requires converting the trained 
// model to a C array, typically using TensorFlow Lite.
const float encoder_w1[NUM_FEATURES][32] = {{0}};
const float encoder_b1[32] = {{0}};
const float encoder_w2[32][16] = {{0}};
const float encoder_b2[16] = {{0}};
// ... and so on for all layers.

// --- Type Definitions ---
typedef enum {{
    RISK_LOW = 0,
    RISK_MEDIUM = 1,
    RISK_HIGH = 2,
    RISK_CRITICAL = 3
}} RiskLevel_t;

typedef struct {{
    uint8_t is_anomaly;
    float reconstruction_error;
    RiskLevel_t risk_level;
    uint32_t timestamp;
}} DetectionResult_t;

// --- Function Prototypes ---
void extract_features_mcu(float* signal, int length, float* features);
float predict_anomaly_mcu(float* features);
DetectionResult_t detect_bearing_fault(float* vibration_signal, int signal_length);

// --- Main Application Logic ---
DetectionResult_t detect_bearing_fault(float* vibration_signal, int signal_length) {{
    DetectionResult_t result;
    float features[NUM_FEATURES];
    
    // Step 1: Extract features from the raw signal
    extract_features_mcu(vibration_signal, signal_length, features);
    
    // Step 2: Get a prediction from the autoencoder model
    result.reconstruction_error = predict_anomaly_mcu(features);
    
    // Step 3: Determine if the error exceeds the threshold
    result.is_anomaly = (result.reconstruction_error > RECONSTRUCTION_THRESHOLD) ? 1 : 0;
    result.timestamp = HAL_GetTick();
    
    // Step 4: Classify the risk level
    if (result.reconstruction_error < RECONSTRUCTION_THRESHOLD * 0.5f) {{
        result.risk_level = RISK_LOW;
    }} else if (result.reconstruction_error < RECONSTRUCTION_THRESHOLD) {{
        result.risk_level = RISK_MEDIUM;
    }} else if (result.reconstruction_error < RECONSTRUCTION_THRESHOLD * 2.0f) {{
        result.risk_level = RISK_HIGH;
    }} else {{
        result.risk_level = RISK_CRITICAL;
    }}
    
    return result;
}}

// --- Optimized MCU Feature Extraction ---
void extract_features_mcu(float* signal, int length, float* features) {{
    // This is a simplified, MCU-friendly version of the Python feature extraction.
    // It uses single-pass calculations to save memory and CPU cycles.
    float sum = 0, sum_sq = 0, sum_abs = 0;
    float min_val = signal[0], max_val = signal[0];
    
    for (int i = 0; i < length; i++) {{
        float val = signal[i];
        sum += val;
        sum_sq += val * val;
        sum_abs += fabsf(val);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }}
    
    float mean = sum / length;
    float variance = (sum_sq / length) - (mean * mean);
    float std_dev = sqrtf(variance > 0 ? variance : 0);
    float rms = sqrtf(sum_sq / length);
    float peak = fmaxf(fabsf(min_val), fabsf(max_val));
    float mean_abs = sum_abs / length;

    // NOTE: The order here MUST match the feature mapping above.
    features[0] = rms;
    features[1] = peak;
    features[2] = (rms > 1e-6f) ? peak / rms : 0.0f; // crest_factor
    // ... Kurtosis and other complex features are approximated or omitted for MCU.
    features[3] = 0.0f; // kurtosis placeholder
    features[4] = 0.0f; // spectral_centroid placeholder
    // ... continue for all 16 features, using simplified calculations.
}}

// --- Simplified Model Inference ---
float predict_anomaly_mcu(float* features) {{
    float scaled_features[NUM_FEATURES];
    
    // Step 1: Scale the features
    for (int i = 0; i < NUM_FEATURES; i++) {{
        scaled_features[i] = (features[i] - scaler_mean[i]) / (scaler_scale[i] + 1e-6f);
    }}
    
    // Step 2: Perform model forward pass (manual implementation)
    // This involves matrix multiplications for each layer (e.g., Dense -> BN -> ReLU).
    // This part is complex and is why TFLM is recommended.
    float reconstructed[NUM_FEATURES] = {{0}}; // Placeholder for decoded output
    
    // Step 3: Calculate reconstruction error (MSE)
    float error = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {{
        float diff = scaled_features[i] - reconstructed[i];
        error += diff * diff;
    }}
    
    return error / NUM_FEATURES;
}}
"""
        with open(output_path, 'w') as f:
            f.write(c_code)
        print(f"STM32 C code stub generated: {output_path}")

# --- Main Demonstration Function ---
def demonstrate_with_nasa_data():
    """Main function to run the entire pipeline: load, train, evaluate, and save."""
    # --- IMPORTANT: UPDATE THIS PATH TO YOUR DATASET LOCATION ---
    dataset_path = "D:/errorDetection"
    
    if not os.path.exists(dataset_path) or not any(Path(dataset_path).iterdir()):
        print(f"Dataset path not found or empty: {dataset_path}")
        print("Please download the NASA IMS Bearing dataset and update the 'dataset_path' variable.")
        return None
    
    detector = NASABearingFaultDetector(dataset_path, test_sets=[1, 2]) # Using sets 1 & 2 for a quicker demo
    try:
        detector.load_and_prepare_data(max_files_per_set=None)
        detector.train_model()
        detector.evaluate_model()
        detector.generate_stm32_code()
        return detector
    except Exception as e:
        print(f"An error occurred during the process: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    detector = demonstrate_with_nasa_data()
    
    if detector and detector.autoencoder.autoencoder:
        print("\n--- Saving Final Model and Supplementary Files ---")
        model_path = "nasa_bearing_autoencoder.keras"
        scaler_path = "scaler.pkl"
        threshold_path = "threshold.npy"
        
        # Save the complete Keras model
        detector.autoencoder.autoencoder.save(model_path)
        
        # Save the scaler object using pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(detector.autoencoder.scaler, f)
            
        # Save the calculated anomaly threshold
        np.save(threshold_path, detector.autoencoder.threshold)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Scaler saved to: {scaler_path}")
        print(f"✅ Threshold saved to: {threshold_path}")