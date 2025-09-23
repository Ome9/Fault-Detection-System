import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
from scipy import signal as sig, stats
from scipy.fft import fft, fftfreq

# --- COPIED FROM TRAINING SCRIPT FOR IDENTICAL FEATURE EXTRACTION ---
# This class is now included to ensure features are calculated correctly.
class NASABearingFeatureExtractor:
    def __init__(self, sampling_rate=20000, target_features=16):
        self.fs = sampling_rate
        self.target_features = target_features

    def extract_time_domain_features(self, signal_data):
        features = {}
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak'] = np.max(np.abs(signal_data))
        mean_abs = np.mean(np.abs(signal_data))
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
        features['kurtosis'] = stats.kurtosis(signal_data)
        features['std'] = np.std(signal_data)
        features['skewness'] = stats.skew(signal_data)
        features['impulse_factor'] = features['peak'] / mean_abs if mean_abs > 0 else 0
        features['energy'] = np.sum(signal_data**2)
        return features

    def extract_frequency_domain_features(self, signal_data):
        features = {}
        fft_vals = fft(signal_data)
        fft_magnitude = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = fftfreq(len(signal_data), 1/self.fs)[:len(fft_vals)//2]
        total_power = np.sum(fft_magnitude**2)
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0
        features['spectral_kurtosis'] = stats.kurtosis(fft_magnitude)
        bearing_bands = {'high_freq_1': (100, 500), 'high_freq_2': (500, 1500), 'bearing_1x': (20, 50), 'bearing_2x': (60, 90)}
        for band_name, (low, high) in bearing_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            if np.any(band_mask):
                band_power = np.sum(fft_magnitude[band_mask]**2)
                features[f'{band_name}_power'] = band_power / total_power if total_power > 0 else 0
            else:
                features[f'{band_name}_power'] = 0
        features['dominant_frequency'] = freqs[np.argmax(fft_magnitude)]
        return features

    def extract_wavelet_features(self, signal_data):
        features = {}
        sos_high = sig.butter(4, 100, btype='high', fs=self.fs, output='sos')
        high_freq = sig.sosfilt(sos_high, signal_data)
        features['wavelet_energy_ratio'] = np.sum(high_freq**2) / np.sum(signal_data**2) if np.sum(signal_data**2) > 0 else 0
        return features

    def extract_optimized_features(self, signal_data):
        time_features = self.extract_time_domain_features(signal_data)
        freq_features = self.extract_frequency_domain_features(signal_data)
        wavelet_features = self.extract_wavelet_features(signal_data)
        all_features = {**time_features, **freq_features, **wavelet_features}
        selected_feature_keys = ['rms', 'peak', 'crest_factor', 'kurtosis', 'spectral_centroid', 'spectral_kurtosis', 'high_freq_1_power', 'high_freq_2_power', 'bearing_1x_power', 'bearing_2x_power', 'dominant_frequency', 'wavelet_energy_ratio', 'std', 'skewness', 'impulse_factor', 'energy']
        feature_vector = [all_features.get(key, 0.0) for key in selected_feature_keys]
        return np.array(feature_vector)

# --- CORRECTED DATA LOADER ---
# This now uses the Feature Extractor class to generate real features.
class NASABearingDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.feature_extractor = NASABearingFeatureExtractor()

    def load_all_data_for_calibration(self, max_files_per_set=200):
        all_features = []
        test_dir = next(self.dataset_path.glob("*1st_test*"), None)
        if not test_dir:
             raise FileNotFoundError(f"Could not find the '1st_test' directory in {self.dataset_path}")

        # Use files from early in the experiment, which are known to be 'normal'
        files = sorted(list(test_dir.glob("2003*")))[:max_files_per_set]
        
        print(f"Loading and processing {len(files)} files for calibration data...")
        for filepath in files:
            try:
                # Load all channels from a file
                data_channels = np.loadtxt(filepath)
                # Process each channel (each is a sample)
                for i in range(data_channels.shape[1]):
                    signal_data = data_channels[:, i]
                    # ADDED: Extract real features using the correct class
                    features = self.feature_extractor.extract_optimized_features(signal_data)
                    all_features.append(features)
            except Exception as e:
                print(f"Skipping file {filepath.name} due to error: {e}")
                continue
        
        return np.array(all_features)


def main():
    """Main function to load, convert, and quantize the trained model."""
    # --- 1. DEFINE FILE PATHS ---
    # !!! UPDATE THIS PATH !!!
    DATASET_PATH = "D:/errorDetection"
    
    SAVED_MODEL_PATH = "nasa_bearing_autoencoder.keras"
    SCALER_PATH = "scaler.pkl"
    TFLITE_MODEL_OUTPUT_PATH = "nasa_bearing_model_quant.tflite"

    print(f"Loading Keras model from: {SAVED_MODEL_PATH}")
    model = tf.keras.models.load_model(SAVED_MODEL_PATH)

    print(f"Loading scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # --- 2. CREATE REPRESENTATIVE DATASET FOR QUANTIZATION ---
    print("\nCreating representative dataset from real data...")
    data_loader = NASABearingDataLoader(DATASET_PATH)
    X_normal = data_loader.load_all_data_for_calibration()
    X_normal_scaled = scaler.transform(X_normal)
    representative_data = X_normal_scaled.astype(np.float32)
    print(f"Successfully created representative dataset with shape: {representative_data.shape}")

    def representative_dataset_gen():
      for i in range(len(representative_data)):
        yield [np.array([representative_data[i]])]

    # --- 3. CONVERT AND QUANTIZE THE MODEL ---
    print("\nStarting model conversion to TensorFlow Lite (int8)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()
    
    # --- 4. SAVE THE FINAL MODEL ---
    with open(TFLITE_MODEL_OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model_quant)
        
    print("\n✅ Success!")
    print(f"Quantized TensorFlow Lite model saved to: {TFLITE_MODEL_OUTPUT_PATH}")
    print(f"\nNext step: Convert '{TFLITE_MODEL_OUTPUT_PATH}' to a C array using the 'xxd' command.")

if __name__ == '__main__':
    main()