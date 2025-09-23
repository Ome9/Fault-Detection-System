/*
 * NASA IMS Bearing Fault Detection System for STM32 F446RE
 * Generated on: 2025-09-04 03:26:41.309991
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
#define NUM_FEATURES             16
#define ENCODING_DIM             8
#define RECONSTRUCTION_THRESHOLD 0.05044226f

// --- Feature Index Mapping (from Python script) ---
// features[0]: rms
// features[1]: peak
// features[2]: crest_factor
// features[3]: kurtosis
// features[4]: spectral_centroid
// features[5]: spectral_kurtosis
// features[6]: high_freq_1_power
// features[7]: high_freq_2_power
// features[8]: bearing_1x_power
// features[9]: bearing_2x_power
// features[10]: envelope_spectral_kurtosis
// features[11]: envelope_peak
// features[12]: std
// features[13]: wavelet_energy_ratio
// features[14]: impulse_factor
// features[15]: energy
// ----------------------------------------------------

// --- Preprocessing Parameters (Scaler) ---
const float scaler_mean[NUM_FEATURES] = {0.12961213f, 0.59487095f, 4.66761920f, 0.75553284f, 4419.08245392f, 6193.57987402f, 0.02727767f, 0.09119051f, 0.00523083f, 0.00147729f, 247.49335756f, 194.12208865f, 0.08861476f, 0.53059210f, 5.64558361f, 362.56943295f};
const float scaler_scale[NUM_FEATURES] = {0.03007127f, 0.15236891f, 1.02252641f, 0.56191098f, 274.71648900f, 3215.74126355f, 0.02554463f, 0.08194032f, 0.00887683f, 0.00197852f, 359.72100928f, 121.51884587f, 0.01564365f, 0.25642719f, 1.42652745f, 140.80424301f};

// --- Model Parameters (Placeholders) ---
// NOTE: These weights are placeholders. A real deployment requires converting the trained 
// model to a C array, typically using TensorFlow Lite.
const float encoder_w1[NUM_FEATURES][32] = {0};
const float encoder_b1[32] = {0};
const float encoder_w2[32][16] = {0};
const float encoder_b2[16] = {0};
// ... and so on for all layers.

// --- Type Definitions ---
typedef enum {
    RISK_LOW = 0,
    RISK_MEDIUM = 1,
    RISK_HIGH = 2,
    RISK_CRITICAL = 3
} RiskLevel_t;

typedef struct {
    uint8_t is_anomaly;
    float reconstruction_error;
    RiskLevel_t risk_level;
    uint32_t timestamp;
} DetectionResult_t;

// --- Function Prototypes ---
void extract_features_mcu(float* signal, int length, float* features);
float predict_anomaly_mcu(float* features);
DetectionResult_t detect_bearing_fault(float* vibration_signal, int signal_length);

// --- Main Application Logic ---
DetectionResult_t detect_bearing_fault(float* vibration_signal, int signal_length) {
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
    if (result.reconstruction_error < RECONSTRUCTION_THRESHOLD * 0.5f) {
        result.risk_level = RISK_LOW;
    } else if (result.reconstruction_error < RECONSTRUCTION_THRESHOLD) {
        result.risk_level = RISK_MEDIUM;
    } else if (result.reconstruction_error < RECONSTRUCTION_THRESHOLD * 2.0f) {
        result.risk_level = RISK_HIGH;
    } else {
        result.risk_level = RISK_CRITICAL;
    }
    
    return result;
}

// --- Optimized MCU Feature Extraction ---
void extract_features_mcu(float* signal, int length, float* features) {
    // This is a simplified, MCU-friendly version of the Python feature extraction.
    // It uses single-pass calculations to save memory and CPU cycles.
    float sum = 0, sum_sq = 0, sum_abs = 0;
    float min_val = signal[0], max_val = signal[0];
    
    for (int i = 0; i < length; i++) {
        float val = signal[i];
        sum += val;
        sum_sq += val * val;
        sum_abs += fabsf(val);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
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
}

// --- Simplified Model Inference ---
float predict_anomaly_mcu(float* features) {
    float scaled_features[NUM_FEATURES];
    
    // Step 1: Scale the features
    for (int i = 0; i < NUM_FEATURES; i++) {
        scaled_features[i] = (features[i] - scaler_mean[i]) / (scaler_scale[i] + 1e-6f);
    }
    
    // Step 2: Perform model forward pass (manual implementation)
    // This involves matrix multiplications for each layer (e.g., Dense -> BN -> ReLU).
    // This part is complex and is why TFLM is recommended.
    float reconstructed[NUM_FEATURES] = {0}; // Placeholder for decoded output
    
    // Step 3: Calculate reconstruction error (MSE)
    float error = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        float diff = scaled_features[i] - reconstructed[i];
        error += diff * diff;
    }
    
    return error / NUM_FEATURES;
}
