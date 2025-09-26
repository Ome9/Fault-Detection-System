/*
 * TinyML NASA Bearing Fault Detection for STM32
 * Optimized deployment with TensorFlow Lite Micro
 * 
 * Features:
 * - 8-feature optimized extraction
 * - INT8 quantized inference
 * - Memory-efficient implementation
 * - Real-time processing capability
 * 
 * Target: STM32 F446RE (or similar ARM Cortex-M4)
 * Author: AI Assistant
 * Date: 2025
 */

#include "stm32f4xx_hal.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define SAMPLE_RATE_HZ          2000    // Downsampled for STM32
#define WINDOW_SIZE             1024    // Processing window
#define NUM_FEATURES            8       // Optimized feature count
#define TENSOR_ARENA_SIZE       8192    // TensorFlow Lite arena size (8KB)
#define ANOMALY_THRESHOLD_FIXED 819     // Fixed-point threshold (0.05 * 16384)
#define FIXED_POINT_SHIFT       14      // Q14 fixed-point format

// Feature indices (must match Python training order)
typedef enum {
    FEATURE_RMS = 0,
    FEATURE_PEAK = 1,
    FEATURE_CREST_FACTOR = 2,
    FEATURE_KURTOSIS = 3,
    FEATURE_ENVELOPE_PEAK = 4,
    FEATURE_HIGH_FREQ_POWER = 5,
    FEATURE_BEARING_FREQ_POWER = 6,
    FEATURE_SPECTRAL_KURTOSIS = 7
} feature_index_t;

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

// TensorFlow Lite Micro components
static tflite::MicroErrorReporter micro_error_reporter;
static tflite::AllOpsResolver resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// Memory allocation for TensorFlow Lite
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Feature scaling parameters (from Python training - convert to fixed-point)
// These values should be updated from your trained model
static const int32_t scaler_mean_fixed[NUM_FEATURES] = {
    2124,   // rms mean * 16384
    9748,   // peak mean * 16384  
    76459,  // crest_factor mean * 16384
    12378,  // kurtosis mean * 16384
    3181,   // envelope_peak mean * 16384
    446,    // high_freq_power mean * 16384
    86,     // bearing_freq_power mean * 16384
    101596  // spectral_kurtosis mean * 16384
};

static const int32_t scaler_scale_fixed[NUM_FEATURES] = {
    492,    // rms scale * 16384
    2496,   // peak scale * 16384
    16753,  // crest_factor scale * 16384
    9205,   // kurtosis scale * 16384
    1991,   // envelope_peak scale * 16384
    418,    // high_freq_power scale * 16384
    32,     // bearing_freq_power scale * 16384
    52682   // spectral_kurtosis scale * 16384
};

// Buffers for signal processing
static float signal_buffer[WINDOW_SIZE];
static int32_t features_fixed[NUM_FEATURES];
static float features_float[NUM_FEATURES];

// =============================================================================
// EXTERNAL MODEL DATA
// =============================================================================

// This should be generated from your .tflite file using xxd or similar tool
// Example: xxd -i advanced_qat_model.tflite > model_data.h
extern const unsigned char model_data[];
extern const unsigned int model_data_len;

// =============================================================================
// FIXED-POINT ARITHMETIC UTILITIES
// =============================================================================

static inline int32_t float_to_fixed(float f) {
    return (int32_t)(f * (1 << FIXED_POINT_SHIFT));
}

static inline float fixed_to_float(int32_t f) {
    return (float)f / (1 << FIXED_POINT_SHIFT);
}

static inline int32_t fixed_multiply(int32_t a, int32_t b) {
    return (int64_t)a * b >> FIXED_POINT_SHIFT;
}

static inline int32_t fixed_divide(int32_t a, int32_t b) {
    return ((int64_t)a << FIXED_POINT_SHIFT) / b;
}

// =============================================================================
// OPTIMIZED FEATURE EXTRACTION
// =============================================================================

/**
 * Fast square root using Newton-Raphson method
 */
static float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    
    // Initial guess using bit manipulation
    union { float f; uint32_t i; } u;
    u.f = x;
    u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
    
    // Two Newton-Raphson iterations
    u.f = 0.5f * (u.f + x / u.f);
    u.f = 0.5f * (u.f + x / u.f);
    
    return u.f;
}

/**
 * Extract time-domain features (RMS, Peak, Crest Factor)
 */
static void extract_time_features(const float* signal, int length) {
    float sum_sq = 0.0f;
    float peak = 0.0f;
    float sum_quad = 0.0f;  // For kurtosis approximation
    
    // Single pass through data
    for (int i = 0; i < length; i++) {
        float abs_val = fabsf(signal[i]);
        float sq_val = signal[i] * signal[i];
        
        sum_sq += sq_val;
        sum_quad += sq_val * sq_val;  // x^4 for kurtosis
        
        if (abs_val > peak) {
            peak = abs_val;
        }
    }
    
    // Calculate features
    float rms = fast_sqrt(sum_sq / length);
    float kurtosis_approx = (sum_quad / length) / ((sum_sq / length) * (sum_sq / length)) - 3.0f;
    
    // Store in fixed-point
    features_fixed[FEATURE_RMS] = float_to_fixed(rms);
    features_fixed[FEATURE_PEAK] = float_to_fixed(peak);
    features_fixed[FEATURE_CREST_FACTOR] = (rms > 1e-6f) ? 
        fixed_divide(float_to_fixed(peak), float_to_fixed(rms)) : 0;
    features_fixed[FEATURE_KURTOSIS] = float_to_fixed(kurtosis_approx);
}

/**
 * Extract envelope peak using Hilbert transform approximation
 */
static void extract_envelope_features(const float* signal, int length) {
    float max_envelope = 0.0f;
    
    // Simplified envelope detection using moving average of absolute values
    int window = 16;  // Small window for efficiency
    
    for (int i = window; i < length - window; i++) {
        float local_avg = 0.0f;
        
        for (int j = -window/2; j <= window/2; j++) {
            local_avg += fabsf(signal[i + j]);
        }
        
        local_avg /= window;
        
        if (local_avg > max_envelope) {
            max_envelope = local_avg;
        }
    }
    
    features_fixed[FEATURE_ENVELOPE_PEAK] = float_to_fixed(max_envelope);
}

/**
 * Extract frequency-domain features using simplified FFT approach
 */
static void extract_frequency_features(const float* signal, int length) {
    // Simplified frequency analysis using filtering
    float high_freq_energy = 0.0f;
    float bearing_freq_energy = 0.0f;
    float spectral_variance = 0.0f;
    
    // Simple high-pass filter for high frequency content
    for (int i = 1; i < length - 1; i++) {
        float high_pass = signal[i] - 0.5f * (signal[i-1] + signal[i+1]);
        high_freq_energy += high_pass * high_pass;
        
        // Band-pass for bearing frequencies (very simplified)
        if (i % 10 == 0) {  // Approximate bearing frequency sampling
            bearing_freq_energy += signal[i] * signal[i];
        }
        
        // Spectral variance approximation
        float diff = signal[i] - signal[i-1];
        spectral_variance += diff * diff;
    }
    
    // Normalize by total energy
    float total_energy = 0.0f;
    for (int i = 0; i < length; i++) {
        total_energy += signal[i] * signal[i];
    }
    
    if (total_energy > 1e-6f) {
        high_freq_energy /= total_energy;
        bearing_freq_energy /= total_energy;
        spectral_variance /= total_energy;
    }
    
    features_fixed[FEATURE_HIGH_FREQ_POWER] = float_to_fixed(high_freq_energy);
    features_fixed[FEATURE_BEARING_FREQ_POWER] = float_to_fixed(bearing_freq_energy);
    features_fixed[FEATURE_SPECTRAL_KURTOSIS] = float_to_fixed(spectral_variance);
}

/**
 * Main feature extraction function
 */
static void extract_all_features(const float* signal, int length) {
    extract_time_features(signal, length);
    extract_envelope_features(signal, length);
    extract_frequency_features(signal, length);
}

/**
 * Scale features using trained scaler parameters
 */
static void scale_features(void) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        // Apply scaling: (feature - mean) / scale
        int32_t centered = features_fixed[i] - scaler_mean_fixed[i];
        features_fixed[i] = fixed_divide(centered, scaler_scale_fixed[i]);
        
        // Convert to float for TensorFlow Lite input
        features_float[i] = fixed_to_float(features_fixed[i]);
    }
}

// =============================================================================
// TENSORFLOW LITE MICRO INTEGRATION
// =============================================================================

/**
 * Initialize TensorFlow Lite Micro interpreter
 */
static bool init_tflite(void) {
    // Load model from flash memory
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\n");
        return false;
    }
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors() failed!\n");
        return false;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Verify tensor dimensions
    if (input->dims->data[1] != NUM_FEATURES) {
        printf("Input tensor dimension mismatch!\n");
        return false;
    }
    
    printf("TensorFlow Lite Micro initialized successfully\n");
    printf("Input shape: [%d, %d]\n", input->dims->data[0], input->dims->data[1]);
    printf("Output shape: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);
    printf("Arena size used: %d bytes\n", TENSOR_ARENA_SIZE);
    
    return true;
}

/**
 * Run inference on extracted features
 */
static float run_inference(void) {
    // Copy features to input tensor
    if (input->type == kTfLiteInt8) {
        // Quantized input
        int8_t* input_data = input->data.int8;
        for (int i = 0; i < NUM_FEATURES; i++) {
            // Convert float to quantized int8
            // This assumes input quantization parameters from your model
            input_data[i] = (int8_t)(features_float[i] * 128.0f);  // Adjust scaling as needed
        }
    } else {
        // Float input
        float* input_data = input->data.f;
        memcpy(input_data, features_float, NUM_FEATURES * sizeof(float));
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("Invoke() failed!\n");
        return -1.0f;
    }
    
    // Get reconstruction error
    float reconstruction_error = 0.0f;
    
    if (output->type == kTfLiteInt8) {
        // Quantized output
        int8_t* output_data = output->data.int8;
        for (int i = 0; i < NUM_FEATURES; i++) {
            float reconstructed = (float)output_data[i] / 128.0f;  // Adjust scaling
            float diff = features_float[i] - reconstructed;
            reconstruction_error += diff * diff;
        }
    } else {
        // Float output
        float* output_data = output->data.f;
        for (int i = 0; i < NUM_FEATURES; i++) {
            float diff = features_float[i] - output_data[i];
            reconstruction_error += diff * diff;
        }
    }
    
    return reconstruction_error / NUM_FEATURES;  // Mean squared error
}

// =============================================================================
// MAIN FAULT DETECTION PIPELINE
// =============================================================================

/**
 * Process signal and detect anomalies
 */
static int detect_bearing_fault(const float* signal, int length) {
    // Step 1: Extract features
    extract_all_features(signal, length);
    
    // Step 2: Scale features
    scale_features();
    
    // Step 3: Run inference
    float reconstruction_error = run_inference();
    
    if (reconstruction_error < 0) {
        return -1;  // Error in inference
    }
    
    // Step 4: Compare with threshold
    float threshold = fixed_to_float(ANOMALY_THRESHOLD_FIXED);
    
    // Debug output
    printf("Reconstruction error: %.6f, Threshold: %.6f\n", 
           reconstruction_error, threshold);
    
    return (reconstruction_error > threshold) ? 1 : 0;  // 1 = anomaly, 0 = normal
}

/**
 * Initialize the bearing fault detection system
 */
bool bearing_fault_detector_init(void) {
    printf("Initializing TinyML Bearing Fault Detector...\n");
    
    // Initialize TensorFlow Lite Micro
    if (!init_tflite()) {
        printf("Failed to initialize TensorFlow Lite Micro\n");
        return false;
    }
    
    printf("Bearing fault detector initialized successfully\n");
    return true;
}

/**
 * Main processing function - call this with new sensor data
 */
int process_sensor_data(const float* sensor_data, int data_length) {
    if (data_length < WINDOW_SIZE) {
        printf("Insufficient data length: %d (need %d)\n", data_length, WINDOW_SIZE);
        return -1;
    }
    
    // Use the first WINDOW_SIZE samples
    return detect_bearing_fault(sensor_data, WINDOW_SIZE);
}

// =============================================================================
// EXAMPLE USAGE AND TESTING
// =============================================================================

/**
 * Test function with synthetic data
 */
void test_fault_detection(void) {
    printf("Testing fault detection with synthetic data...\n");
    
    // Generate test signal
    for (int i = 0; i < WINDOW_SIZE; i++) {
        // Normal bearing signal (random noise)
        signal_buffer[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    int result = detect_bearing_fault(signal_buffer, WINDOW_SIZE);
    printf("Normal signal result: %d (0=normal, 1=fault)\n", result);
    
    // Generate faulty signal (with periodic impulses)
    for (int i = 0; i < WINDOW_SIZE; i++) {
        signal_buffer[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        
        // Add periodic impulses
        if (i % 100 == 0) {
            signal_buffer[i] += 1.0f;  // Large impulse
        }
    }
    
    result = detect_bearing_fault(signal_buffer, WINDOW_SIZE);
    printf("Faulty signal result: %d (0=normal, 1=fault)\n", result);
}

/**
 * Print system information
 */
void print_system_info(void) {
    printf("\\n=== TinyML Bearing Fault Detector ===\\n");
    printf("Target: STM32 F446RE\\n");
    printf("Sample rate: %d Hz\\n", SAMPLE_RATE_HZ);
    printf("Window size: %d samples\\n", WINDOW_SIZE);
    printf("Features: %d (optimized)\\n", NUM_FEATURES);
    printf("Model: INT8 quantized autoencoder\\n");
    printf("Memory usage: %d bytes (tensor arena)\\n", TENSOR_ARENA_SIZE);
    printf("Fixed-point format: Q%d\\n", FIXED_POINT_SHIFT);
    printf("=====================================\\n\\n");
}

// =============================================================================
// MAIN FUNCTION (FOR TESTING)
// =============================================================================

int main(void) {
    // Initialize system
    HAL_Init();
    
    // Print system information
    print_system_info();
    
    // Initialize fault detector
    if (!bearing_fault_detector_init()) {
        printf("Initialization failed!\\n");
        return -1;
    }
    
    // Run tests
    test_fault_detection();
    
    // Main processing loop
    printf("Entering main processing loop...\\n");
    while (1) {
        // In a real application, you would:
        // 1. Acquire sensor data from ADC
        // 2. Fill signal_buffer with new data
        // 3. Call process_sensor_data()
        // 4. Act on the result (LED, alarm, etc.)
        
        HAL_Delay(1000);  // Process every second
        
        // Example: process synthetic data
        // int fault_detected = process_sensor_data(signal_buffer, WINDOW_SIZE);
        // Handle fault detection result...
    }
    
    return 0;
}