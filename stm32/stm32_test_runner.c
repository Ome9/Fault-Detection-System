/*
 * Test Runner for STM32 TinyML Bearing Fault Detection
 * Validates the system using synthetic test data without physical sensors
 * 
 * This file demonstrates how to run validation tests on your STM32
 * using the pre-generated test vectors
 * 
 * Target: STM32 F446RE (or similar ARM Cortex-M4)
 * Author: AI Assistant
 * Date: 2025-09-26
 */

#include "stm32f4xx_hal.h"
#include "stm32_test_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

// =============================================================================
// EXTERNAL DECLARATIONS
// =============================================================================

// Model data (include the generated model_data.h file)
extern const unsigned char model_data[];
extern const unsigned int model_data_len;

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
#define TENSOR_ARENA_SIZE 8192
alignas(16) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// Test results tracking
typedef struct {
    uint32_t total_tests;
    uint32_t passed_tests;
    uint32_t failed_tests;
    uint32_t true_positives;   // Correctly detected faults
    uint32_t true_negatives;   // Correctly detected normal
    uint32_t false_positives;  // False alarms
    uint32_t false_negatives;  // Missed faults
} test_results_t;

static test_results_t test_results = {0};

// =============================================================================
// INITIALIZATION FUNCTIONS
// =============================================================================

/**
 * Initialize TensorFlow Lite Micro model
 */
HAL_StatusTypeDef init_tflite_model(void) {
    // Load model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version %d not supported. Expected version %d\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return HAL_ERROR;
    }
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, &micro_error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        return HAL_ERROR;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Verify tensor dimensions
    if (input->dims->size != 2 || input->dims->data[1] != NUM_FEATURES) {
        printf("Input tensor dimensions incorrect. Expected [1, %d], got [", NUM_FEATURES);
        for (int i = 0; i < input->dims->size; i++) {
            printf("%d", input->dims->data[i]);
            if (i < input->dims->size - 1) printf(", ");
        }
        printf("]\n");
        return HAL_ERROR;
    }
    
    printf("TensorFlow Lite model initialized successfully\n");
    printf("Input tensor: [%d, %d]\n", input->dims->data[0], input->dims->data[1]);
    printf("Output tensor: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);
    printf("Tensor arena usage: %d bytes\n", TENSOR_ARENA_SIZE);
    
    return HAL_OK;
}

// =============================================================================
// INFERENCE FUNCTIONS
// =============================================================================

/**
 * Run inference on scaled features
 * Returns reconstruction error as float
 */
float run_inference(const float* scaled_features) {
    // Copy input data
    for (int i = 0; i < NUM_FEATURES; i++) {
        input->data.f[i] = scaled_features[i];
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("Invoke failed\n");
        return -1.0f;
    }
    
    // Calculate reconstruction error (MSE)
    float reconstruction_error = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        float diff = input->data.f[i] - output->data.f[i];
        reconstruction_error += diff * diff;
    }
    reconstruction_error /= NUM_FEATURES;
    
    return reconstruction_error;
}

// =============================================================================
// TEST EXECUTION FUNCTIONS
// =============================================================================

/**
 * Run a single test case
 */
HAL_StatusTypeDef run_single_test(int test_index) {
    const test_case_t* test_case = get_test_case(test_index);
    if (test_case == NULL) {
        printf("Invalid test index: %d\n", test_index);
        return HAL_ERROR;
    }
    
    // Scale features for model input
    float scaled_features[NUM_FEATURES];
    scale_features(test_case->features, scaled_features);
    
    // Run inference
    float reconstruction_error = run_inference(scaled_features);
    if (reconstruction_error < 0.0f) {
        printf("Inference failed for test %d\n", test_index);
        return HAL_ERROR;
    }
    
    // Determine prediction
    uint8_t predicted_label = is_fault_detected(reconstruction_error);
    uint8_t actual_label = test_case->expected_label;
    
    // Update statistics
    test_results.total_tests++;
    if (predicted_label == actual_label) {
        test_results.passed_tests++;
        if (actual_label == LABEL_NORMAL) {
            test_results.true_negatives++;
        } else {
            test_results.true_positives++;
        }
    } else {
        test_results.failed_tests++;
        if (predicted_label == LABEL_FAULT && actual_label == LABEL_NORMAL) {
            test_results.false_positives++;
        } else {
            test_results.false_negatives++;
        }
    }
    
    // Print detailed results
    printf("\\n=== Test Case %d: %s ===\\n", test_index + 1, test_case_names[test_index]);
    printf("Description: %s\\n", test_case->description);
    printf("Raw Features (Q14): [");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%ld", test_case->features[i]);
        if (i < NUM_FEATURES - 1) printf(", ");
    }
    printf("]\\n");
    
    printf("Scaled Features: [");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%.3f", scaled_features[i]);
        if (i < NUM_FEATURES - 1) printf(", ");
    }
    printf("]\\n");
    
    printf("Expected Label: %s\\n", actual_label == LABEL_NORMAL ? "NORMAL" : "FAULT");
    printf("Predicted Label: %s\\n", predicted_label == LABEL_NORMAL ? "NORMAL" : "FAULT");
    printf("Expected Error: %.6f\\n", test_case->expected_error);
    printf("Actual Error: %.6f\\n", reconstruction_error);
    printf("Threshold: %.6f\\n", (float)ANOMALY_THRESHOLD / (1 << FIXED_POINT_SHIFT));
    printf("Result: %s\\n", predicted_label == actual_label ? "✅ PASS" : "❌ FAIL");
    
    // Performance metrics per test
    if (predicted_label == actual_label) {
        float error_diff = fabs(reconstruction_error - test_case->expected_error);
        printf("Error Accuracy: %.1f%% (diff: %.6f)\\n", 
               (1.0f - error_diff / test_case->expected_error) * 100.0f, error_diff);
    }
    
    return HAL_OK;
}

/**
 * Run all test cases
 */
HAL_StatusTypeDef run_all_tests(void) {
    printf("\\n" "========================================\\n");
    printf("STM32 TinyML Bearing Fault Detection Test Suite\\n");
    printf("========================================\\n");
    printf("Total test cases: %d\\n", NUM_TEST_CASES);
    printf("Normal cases: %d\\n", NUM_NORMAL_CASES);
    printf("Fault cases: %d\\n", NUM_FAULT_CASES);
    printf("Anomaly threshold: %.6f\\n", (float)ANOMALY_THRESHOLD / (1 << FIXED_POINT_SHIFT));
    
    // Reset test results
    memset(&test_results, 0, sizeof(test_results_t));
    
    // Run all tests
    for (int i = 0; i < NUM_TEST_CASES; i++) {
        if (run_single_test(i) != HAL_OK) {
            printf("Test %d failed to execute\\n", i + 1);
            return HAL_ERROR;
        }
        
        // Small delay between tests (optional)
        HAL_Delay(100);
    }
    
    return HAL_OK;
}

/**
 * Print comprehensive test summary
 */
void print_test_summary(void) {
    printf("\\n" "========================================\\n");
    printf("TEST SUMMARY\\n");
    printf("========================================\\n");
    
    // Basic statistics
    printf("Total Tests: %lu\\n", test_results.total_tests);
    printf("Passed: %lu (%.1f%%)\\n", test_results.passed_tests, 
           (float)test_results.passed_tests / test_results.total_tests * 100.0f);
    printf("Failed: %lu (%.1f%%)\\n", test_results.failed_tests,
           (float)test_results.failed_tests / test_results.total_tests * 100.0f);
    
    // Confusion matrix
    printf("\\nConfusion Matrix:\\n");
    printf("                 Predicted\\n");
    printf("                Normal  Fault\\n");
    printf("Actual Normal   %4lu   %4lu\\n", test_results.true_negatives, test_results.false_positives);
    printf("       Fault    %4lu   %4lu\\n", test_results.false_negatives, test_results.true_positives);
    
    // Performance metrics
    float accuracy = (float)(test_results.true_positives + test_results.true_negatives) / test_results.total_tests;
    float precision = test_results.true_positives > 0 ? 
                     (float)test_results.true_positives / (test_results.true_positives + test_results.false_positives) : 0.0f;
    float recall = test_results.true_positives > 0 ? 
                   (float)test_results.true_positives / (test_results.true_positives + test_results.false_negatives) : 0.0f;
    float f1_score = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0f;
    float specificity = test_results.true_negatives > 0 ? 
                        (float)test_results.true_negatives / (test_results.true_negatives + test_results.false_positives) : 0.0f;
    
    printf("\\nPerformance Metrics:\\n");
    printf("Accuracy:    %.1f%%\\n", accuracy * 100.0f);
    printf("Precision:   %.1f%%\\n", precision * 100.0f);
    printf("Recall:      %.1f%% (Sensitivity)\\n", recall * 100.0f);
    printf("Specificity: %.1f%%\\n", specificity * 100.0f);
    printf("F1-Score:    %.1f%%\\n", f1_score * 100.0f);
    
    // Fault detection specific metrics
    printf("\\nFault Detection Analysis:\\n");
    printf("True Positives:  %lu / %d (%.1f%%)\\n", test_results.true_positives, NUM_FAULT_CASES,
           (float)test_results.true_positives / NUM_FAULT_CASES * 100.0f);
    printf("False Negatives: %lu / %d (%.1f%%)\\n", test_results.false_negatives, NUM_FAULT_CASES,
           (float)test_results.false_negatives / NUM_FAULT_CASES * 100.0f);
    printf("False Alarms:    %lu / %d (%.1f%%)\\n", test_results.false_positives, NUM_NORMAL_CASES,
           (float)test_results.false_positives / NUM_NORMAL_CASES * 100.0f);
    
    // Overall assessment
    printf("\\nOverall Assessment:\\n");
    if (accuracy >= 0.9f) {
        printf("🎯 EXCELLENT: System performance is excellent (≥90%% accuracy)\\n");
    } else if (accuracy >= 0.8f) {
        printf("✅ GOOD: System performance is good (≥80%% accuracy)\\n");
    } else if (accuracy >= 0.7f) {
        printf("⚠️  FAIR: System performance is fair (≥70%% accuracy)\\n");
    } else {
        printf("❌ POOR: System performance needs improvement (<70%% accuracy)\\n");
    }
    
    if (test_results.false_negatives == 0) {
        printf("✅ No missed faults - excellent safety performance\\n");
    } else {
        printf("⚠️  %lu missed faults - review detection threshold\\n", test_results.false_negatives);
    }
    
    if (test_results.false_positives <= 1) {
        printf("✅ Low false alarm rate - good operational efficiency\\n");
    } else {
        printf("⚠️  %lu false alarms - consider threshold tuning\\n", test_results.false_positives);
    }
}

// =============================================================================
// MAIN TEST FUNCTION
// =============================================================================

/**
 * Main test execution function
 * Call this from your main() function or test task
 */
HAL_StatusTypeDef run_bearing_fault_tests(void) {
    printf("\\n" "Initializing STM32 TinyML Bearing Fault Detection Tests...\\n");
    
    // Initialize TensorFlow Lite model
    if (init_tflite_model() != HAL_OK) {
        printf("Failed to initialize TensorFlow Lite model\\n");
        return HAL_ERROR;
    }
    
    // Run all tests
    if (run_all_tests() != HAL_OK) {
        printf("Test execution failed\\n");
        return HAL_ERROR;
    }
    
    // Print summary
    print_test_summary();
    
    printf("\\n" "========================================\\n");
    printf("Test execution completed successfully!\\n");
    printf("========================================\\n");
    
    return HAL_OK;
}

// =============================================================================
// INDIVIDUAL TEST FUNCTIONS (for debugging)
// =============================================================================

/**
 * Test only normal bearing cases
 */
HAL_StatusTypeDef test_normal_bearings(void) {
    printf("\\n" "Testing Normal Bearing Cases Only...\\n");
    
    for (int i = 0; i < NUM_NORMAL_CASES; i++) {
        if (run_single_test(i) != HAL_OK) {
            return HAL_ERROR;
        }
    }
    
    return HAL_OK;
}

/**
 * Test only faulty bearing cases
 */
HAL_StatusTypeDef test_faulty_bearings(void) {
    printf("\\n" "Testing Faulty Bearing Cases Only...\\n");
    
    for (int i = NUM_NORMAL_CASES; i < NUM_TEST_CASES; i++) {
        if (run_single_test(i) != HAL_OK) {
            return HAL_ERROR;
        }
    }
    
    return HAL_OK;
}

/**
 * Demo function showing how to test individual cases
 */
void demo_individual_tests(void) {
    printf("\\n" "=== Individual Test Demo ===\\n");
    
    // Test a specific healthy bearing
    printf("\\n" "Testing healthy bearing (Case 1):\\n");
    run_single_test(0);
    
    // Test a specific faulty bearing  
    printf("\\n" "Testing faulty bearing (Case 8):\\n");
    run_single_test(7);
    
    // Test edge case
    printf("\\n" "Testing edge case (Case 6 - end of normal range):\\n");
    run_single_test(5);
}

// =============================================================================
// UTILITY FUNCTIONS FOR INTEGRATION
// =============================================================================

/**
 * Get test results for external analysis
 */
const test_results_t* get_test_results(void) {
    return &test_results;
}

/**
 * Reset test statistics
 */
void reset_test_results(void) {
    memset(&test_results, 0, sizeof(test_results_t));
}

/**
 * Quick validation function for deployment
 * Returns HAL_OK if all tests pass, HAL_ERROR otherwise
 */
HAL_StatusTypeDef validate_system(void) {
    printf("Running quick system validation...\\n");
    
    if (init_tflite_model() != HAL_OK) {
        return HAL_ERROR;
    }
    
    if (run_all_tests() != HAL_OK) {
        return HAL_ERROR;
    }
    
    // Check if validation passes (at least 80% accuracy)
    float accuracy = (float)(test_results.true_positives + test_results.true_negatives) / test_results.total_tests;
    
    if (accuracy >= 0.8f && test_results.false_negatives <= 1) {
        printf("✅ System validation PASSED (Accuracy: %.1f%%)\\n", accuracy * 100.0f);
        return HAL_OK;
    } else {
        printf("❌ System validation FAILED (Accuracy: %.1f%%)\\n", accuracy * 100.0f);
        return HAL_ERROR;
    }
}