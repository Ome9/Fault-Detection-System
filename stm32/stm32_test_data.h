/*
 * Test Data for STM32 TinyML Bearing Fault Detection
 * Generated synthetic test cases based on NASA IMS bearing dataset characteristics
 * 
 * This file contains realistic test vectors for validation without physical sensors
 * 
 * Target: STM32 F446RE (or similar ARM Cortex-M4)
 * Author: AI Assistant  
 * Date: 2025-09-26
 */

#ifndef STM32_TEST_DATA_H
#define STM32_TEST_DATA_H

#include <stdint.h>

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

#define NUM_TEST_CASES      12      // Total test cases
#define NUM_NORMAL_CASES    6       // Normal bearing cases
#define NUM_FAULT_CASES     6       // Faulty bearing cases
#define NUM_FEATURES        8       // 8 optimized features
#define FIXED_POINT_SHIFT   14      // Q14 fixed-point format
#define ANOMALY_THRESHOLD   819     // Fixed-point threshold (0.05 * 16384)

// Test case labels
#define LABEL_NORMAL        0
#define LABEL_FAULT         1

// =============================================================================
// TEST DATA STRUCTURE
// =============================================================================

typedef struct {
    int32_t features[NUM_FEATURES];  // Feature values in Q14 fixed-point
    uint8_t expected_label;          // Expected classification (0=normal, 1=fault)
    const char* description;         // Test case description
    float expected_error;            // Expected reconstruction error
} test_case_t;

// =============================================================================
// REALISTIC TEST VECTORS
// =============================================================================

/*
 * Test data generated based on NASA IMS bearing dataset analysis:
 * - Normal bearings: Low RMS, stable patterns, low kurtosis
 * - Faulty bearings: Higher RMS, increased peaks, elevated kurtosis
 * - Values scaled and converted to Q14 fixed-point format
 */

static const test_case_t test_cases[NUM_TEST_CASES] = {
    
    // =========================
    // NORMAL BEARING TEST CASES  
    // =========================
    
    {
        // Test Case 1: Healthy bearing - early operation
        .features = {
            2048,   // RMS: 0.125 * 16384 (low vibration)
            4096,   // Peak: 0.25 * 16384 (moderate peaks)
            32768,  // Crest Factor: 2.0 * 16384 (normal ratio)
            49152,  // Kurtosis: 3.0 * 16384 (gaussian-like)
            3277,   // Envelope Peak: 0.2 * 16384
            1638,   // High Freq Power: 0.1 * 16384 (low)
            819,    // Bearing Freq Power: 0.05 * 16384 (minimal)
            65536   // Spectral Kurtosis: 4.0 * 16384 (normal)
        },
        .expected_label = LABEL_NORMAL,
        .description = "Healthy bearing - early operation",
        .expected_error = 0.015f
    },
    
    {
        // Test Case 2: Healthy bearing - stable operation  
        .features = {
            2457,   // RMS: 0.15 * 16384
            4915,   // Peak: 0.3 * 16384
            26214,  // Crest Factor: 1.6 * 16384
            45875,  // Kurtosis: 2.8 * 16384
            2949,   // Envelope Peak: 0.18 * 16384
            1311,   // High Freq Power: 0.08 * 16384
            655,    // Bearing Freq Power: 0.04 * 16384
            59507   // Spectral Kurtosis: 3.63 * 16384
        },
        .expected_label = LABEL_NORMAL,
        .description = "Healthy bearing - stable operation",
        .expected_error = 0.012f
    },
    
    {
        // Test Case 3: Healthy bearing - light load
        .features = {
            1843,   // RMS: 0.1125 * 16384
            3686,   // Peak: 0.225 * 16384  
            34611,  // Crest Factor: 2.11 * 16384
            52429,  // Kurtosis: 3.2 * 16384
            2785,   // Envelope Peak: 0.17 * 16384
            983,    // High Freq Power: 0.06 * 16384
            491,    // Bearing Freq Power: 0.03 * 16384
            57016   // Spectral Kurtosis: 3.48 * 16384
        },
        .expected_label = LABEL_NORMAL,
        .description = "Healthy bearing - light load",
        .expected_error = 0.018f
    },
    
    {
        // Test Case 4: Healthy bearing - moderate load
        .features = {
            2621,   // RMS: 0.16 * 16384
            5242,   // Peak: 0.32 * 16384
            30147,  // Crest Factor: 1.84 * 16384
            47186,  // Kurtosis: 2.88 * 16384
            3441,   // Envelope Peak: 0.21 * 16384
            1475,   // High Freq Power: 0.09 * 16384
            737,    // Bearing Freq Power: 0.045 * 16384
            62259   // Spectral Kurtosis: 3.8 * 16384
        },
        .expected_label = LABEL_NORMAL,
        .description = "Healthy bearing - moderate load",
        .expected_error = 0.020f
    },
    
    {
        // Test Case 5: Healthy bearing - optimal conditions
        .features = {
            2048,   // RMS: 0.125 * 16384
            4096,   // Peak: 0.25 * 16384
            32768,  // Crest Factor: 2.0 * 16384
            49152,  // Kurtosis: 3.0 * 16384
            3113,   // Envelope Peak: 0.19 * 16384
            1229,   // High Freq Power: 0.075 * 16384
            614,    // Bearing Freq Power: 0.0375 * 16384
            60293   // Spectral Kurtosis: 3.68 * 16384
        },
        .expected_label = LABEL_NORMAL,
        .description = "Healthy bearing - optimal conditions",
        .expected_error = 0.010f
    },
    
    {
        // Test Case 6: Healthy bearing - end of normal range
        .features = {
            2949,   // RMS: 0.18 * 16384
            5898,   // Peak: 0.36 * 16384
            27525,  // Crest Factor: 1.68 * 16384
            44040,  // Kurtosis: 2.69 * 16384
            3932,   // Envelope Peak: 0.24 * 16384
            1802,   // High Freq Power: 0.11 * 16384
            901,    // Bearing Freq Power: 0.055 * 16384
            68813   // Spectral Kurtosis: 4.2 * 16384
        },
        .expected_label = LABEL_NORMAL,
        .description = "Healthy bearing - end of normal range",
        .expected_error = 0.025f
    },
    
    // =========================
    // FAULTY BEARING TEST CASES
    // =========================
    
    {
        // Test Case 7: Early stage fault - slight degradation
        .features = {
            4096,   // RMS: 0.25 * 16384 (increased)
            9830,   // Peak: 0.6 * 16384 (higher peaks)
            39321,  // Crest Factor: 2.4 * 16384 (elevated)
            65536,  // Kurtosis: 4.0 * 16384 (non-gaussian)
            6554,   // Envelope Peak: 0.4 * 16384 (higher)
            3277,   // High Freq Power: 0.2 * 16384 (increased)
            1638,   // Bearing Freq Power: 0.1 * 16384 (elevated)
            90112   // Spectral Kurtosis: 5.5 * 16384 (abnormal)
        },
        .expected_label = LABEL_FAULT,
        .description = "Early stage fault - slight degradation",
        .expected_error = 0.085f
    },
    
    {
        // Test Case 8: Developing fault - clear symptoms
        .features = {
            5734,   // RMS: 0.35 * 16384 (high vibration)
            13107,  // Peak: 0.8 * 16384 (significant peaks)
            45875,  // Crest Factor: 2.8 * 16384 (high)
            81920,  // Kurtosis: 5.0 * 16384 (impulsive)
            9830,   // Envelope Peak: 0.6 * 16384 (elevated)
            4915,   // High Freq Power: 0.3 * 16384 (high frequency content)
            2458,   // Bearing Freq Power: 0.15 * 16384 (fault frequencies)
            114688  // Spectral Kurtosis: 7.0 * 16384 (highly impulsive)
        },
        .expected_label = LABEL_FAULT,
        .description = "Developing fault - clear symptoms",
        .expected_error = 0.125f
    },
    
    {
        // Test Case 9: Advanced fault - severe symptoms
        .features = {
            8192,   // RMS: 0.5 * 16384 (very high)
            16384,  // Peak: 1.0 * 16384 (extreme peaks)
            52429,  // Crest Factor: 3.2 * 16384 (very high)
            114688, // Kurtosis: 7.0 * 16384 (extremely impulsive)
            13107,  // Envelope Peak: 0.8 * 16384 (severe)
            6554,   // High Freq Power: 0.4 * 16384 (dominant high freq)
            3277,   // Bearing Freq Power: 0.2 * 16384 (strong fault signature)
            147456  // Spectral Kurtosis: 9.0 * 16384 (severe impulsiveness)
        },
        .expected_label = LABEL_FAULT,
        .description = "Advanced fault - severe symptoms",
        .expected_error = 0.220f
    },
    
    {
        // Test Case 10: Inner race fault signature
        .features = {
            6554,   // RMS: 0.4 * 16384
            11469,  // Peak: 0.7 * 16384
            42598,  // Crest Factor: 2.6 * 16384
            98304,  // Kurtosis: 6.0 * 16384 (impulsive inner race)
            8192,   // Envelope Peak: 0.5 * 16384
            5734,   // High Freq Power: 0.35 * 16384
            2867,   // Bearing Freq Power: 0.175 * 16384 (BPFI signature)
            131072  // Spectral Kurtosis: 8.0 * 16384
        },
        .expected_label = LABEL_FAULT,
        .description = "Inner race fault signature",
        .expected_error = 0.165f
    },
    
    {
        // Test Case 11: Outer race fault signature  
        .features = {
            5734,   // RMS: 0.35 * 16384
            10486,  // Peak: 0.64 * 16384
            36700,  // Crest Factor: 2.24 * 16384
            90112,  // Kurtosis: 5.5 * 16384 (outer race characteristic)
            7373,   // Envelope Peak: 0.45 * 16384
            4587,   // High Freq Power: 0.28 * 16384
            2294,   // Bearing Freq Power: 0.14 * 16384 (BPFO signature)
            122880  // Spectral Kurtosis: 7.5 * 16384
        },
        .expected_label = LABEL_FAULT,
        .description = "Outer race fault signature",
        .expected_error = 0.145f
    },
    
    {
        // Test Case 12: Ball/roller fault signature
        .features = {
            7373,   // RMS: 0.45 * 16384  
            14746,  // Peak: 0.9 * 16384
            49152,  // Crest Factor: 3.0 * 16384
            106496, // Kurtosis: 6.5 * 16384 (rolling element characteristic)
            11469,  // Envelope Peak: 0.7 * 16384
            5898,   // High Freq Power: 0.36 * 16384
            2949,   // Bearing Freq Power: 0.18 * 16384 (BSF signature)
            139264  // Spectral Kurtosis: 8.5 * 16384
        },
        .expected_label = LABEL_FAULT,
        .description = "Ball/roller fault signature",
        .expected_error = 0.195f
    }
};

// =============================================================================
// FEATURE SCALING PARAMETERS (Q14 FIXED-POINT)
// =============================================================================

// MinMax scaler parameters converted to fixed-point
// These should match your trained model's scaler
static const int32_t scaler_min_fixed[NUM_FEATURES] = {
    819,    // RMS min: 0.05 * 16384
    1638,   // Peak min: 0.1 * 16384
    16384,  // Crest Factor min: 1.0 * 16384
    32768,  // Kurtosis min: 2.0 * 16384
    655,    // Envelope Peak min: 0.04 * 16384
    327,    // High Freq Power min: 0.02 * 16384
    164,    // Bearing Freq Power min: 0.01 * 16384
    49152   // Spectral Kurtosis min: 3.0 * 16384
};

static const int32_t scaler_max_fixed[NUM_FEATURES] = {
    13107,  // RMS max: 0.8 * 16384
    26214,  // Peak max: 1.6 * 16384
    81920,  // Crest Factor max: 5.0 * 16384
    163840, // Kurtosis max: 10.0 * 16384
    20480,  // Envelope Peak max: 1.25 * 16384
    9830,   // High Freq Power max: 0.6 * 16384
    4915,   // Bearing Freq Power max: 0.3 * 16384
    196608  // Spectral Kurtosis max: 12.0 * 16384
};

// =============================================================================
// EXPECTED OUTPUTS FOR VALIDATION
// =============================================================================

// Expected model outputs (reconstruction errors) for each test case
static const float expected_reconstruction_errors[NUM_TEST_CASES] = {
    0.015f,  // Test case 1 - healthy
    0.012f,  // Test case 2 - healthy
    0.018f,  // Test case 3 - healthy
    0.020f,  // Test case 4 - healthy  
    0.010f,  // Test case 5 - healthy
    0.025f,  // Test case 6 - healthy
    0.085f,  // Test case 7 - early fault
    0.125f,  // Test case 8 - developing fault
    0.220f,  // Test case 9 - advanced fault
    0.165f,  // Test case 10 - inner race fault
    0.145f,  // Test case 11 - outer race fault
    0.195f   // Test case 12 - ball/roller fault
};

// Expected classifications (0=normal, 1=fault)
static const uint8_t expected_classifications[NUM_TEST_CASES] = {
    0, 0, 0, 0, 0, 0,  // Normal cases (1-6)
    1, 1, 1, 1, 1, 1   // Fault cases (7-12)
};

// Test case names for debugging
static const char* test_case_names[NUM_TEST_CASES] = {
    "Healthy_Early",
    "Healthy_Stable", 
    "Healthy_Light",
    "Healthy_Moderate",
    "Healthy_Optimal",
    "Healthy_EndRange",
    "Fault_Early",
    "Fault_Developing",
    "Fault_Advanced",
    "Fault_InnerRace",
    "Fault_OuterRace",
    "Fault_BallRoller"
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Scale features using MinMax scaler (converts to 0-1 range)
 * Input: raw features in Q14 fixed-point
 * Output: scaled features as floats (0.0 to 1.0)
 */
static inline void scale_features(const int32_t* raw_features, float* scaled_features) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        // Convert to float and apply MinMax scaling: (x - min) / (max - min)
        float raw_float = (float)raw_features[i] / (1 << FIXED_POINT_SHIFT);
        float min_float = (float)scaler_min_fixed[i] / (1 << FIXED_POINT_SHIFT);
        float max_float = (float)scaler_max_fixed[i] / (1 << FIXED_POINT_SHIFT);
        
        scaled_features[i] = (raw_float - min_float) / (max_float - min_float);
        
        // Clamp to [0, 1] range
        if (scaled_features[i] < 0.0f) scaled_features[i] = 0.0f;
        if (scaled_features[i] > 1.0f) scaled_features[i] = 1.0f;
    }
}

/**
 * Get test case by index
 */
static inline const test_case_t* get_test_case(int index) {
    if (index < 0 || index >= NUM_TEST_CASES) {
        return NULL;
    }
    return &test_cases[index];
}

/**
 * Check if reconstruction error indicates fault
 */
static inline uint8_t is_fault_detected(float reconstruction_error) {
    return (reconstruction_error * (1 << FIXED_POINT_SHIFT)) > ANOMALY_THRESHOLD ? 1 : 0;
}

#endif // STM32_TEST_DATA_H