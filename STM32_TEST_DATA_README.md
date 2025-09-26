---
marp: true
---

# STM32 TinyML Bearing Fault Detection - Test Data Guide

This guide explains how to use the synthetic test data for validating your STM32 bearing fault detection system without physical sensors.

## 📁 Files Overview

### Core Test Files
- **`stm32_test_data.h`** - Contains 12 realistic test cases with expected results
- **`stm32_test_runner.c`** - Complete test execution and validation framework  
- **`stm32_main_example.c`** - Example integration into your STM32 main application

### Required Dependencies
- **`model_data.h`** - Your TensorFlow Lite model (generated from .tflite file)
- **TensorFlow Lite Micro library** - For model inference
- **STM32 HAL library** - For hardware abstraction

## 🧪 Test Data Characteristics

### Test Cases (12 total)
**Normal Bearings (6 cases):**
1. Healthy bearing - early operation
2. Healthy bearing - stable operation  
3. Healthy bearing - light load
4. Healthy bearing - moderate load
5. Healthy bearing - optimal conditions
6. Healthy bearing - end of normal range

**Faulty Bearings (6 cases):**
7. Early stage fault - slight degradation
8. Developing fault - clear symptoms
9. Advanced fault - severe symptoms
10. Inner race fault signature
11. Outer race fault signature
12. Ball/roller fault signature

### Features (8 per test case)
1. **RMS** - Root Mean Square of vibration amplitude
2. **Peak** - Maximum vibration amplitude
3. **Crest Factor** - Peak to RMS ratio (indicates impulsiveness)
4. **Kurtosis** - Statistical measure of signal impulsiveness
5. **Envelope Peak** - Peak of envelope signal (fault detection)
6. **High Freq Power** - High frequency content power
7. **Bearing Freq Power** - Power at bearing fault frequencies
8. **Spectral Kurtosis** - Frequency domain impulsiveness measure

### Data Format
- **Fixed-point**: Q14 format (14-bit fractional part)
- **Scaling**: MinMax scaling to [0, 1] range for model input
- **Threshold**: 0.05 reconstruction error threshold for fault detection

## 🚀 Quick Start Guide

### Step 1: Include the Test Files
```c
#include "stm32_test_data.h"
// Declare external test functions
extern HAL_StatusTypeDef run_bearing_fault_tests(void);
extern HAL_StatusTypeDef validate_system(void);
```

### Step 2: Initialize Your System
```c
int main(void) {
    // Initialize HAL, clocks, peripherals
    HAL_Init();
    SystemClock_Config();
    UART_Init();  // For debug output
    
    // Run validation
    if (validate_system() == HAL_OK) {
        printf("✅ System validation PASSED\\n");
        // Your application code here
    } else {
        printf("❌ System validation FAILED\\n");
        Error_Handler();
    }
}
```

### Step 3: Run Tests
```c
// Quick validation (recommended for deployment)
HAL_StatusTypeDef status = validate_system();

// Or run full test suite (for development/debugging)
HAL_StatusTypeDef status = run_bearing_fault_tests();
```

## 📊 Expected Results

### Performance Targets
- **Accuracy**: ≥90% (Excellent), ≥80% (Good), ≥70% (Fair)
- **False Negatives**: ≤1 (missed faults - critical for safety)
- **False Positives**: ≤1 (false alarms - affects efficiency)

### Typical Output Example
```
=== Test Case 1: Healthy_Early ===
Description: Healthy bearing - early operation  
Expected Label: NORMAL
Predicted Label: NORMAL
Expected Error: 0.015000
Actual Error: 0.012456
Result: ✅ PASS

========================================
TEST SUMMARY
========================================
Total Tests: 12
Passed: 11 (91.7%)
Failed: 1 (8.3%)

Confusion Matrix:
                 Predicted
                Normal  Fault
Actual Normal      6      0
       Fault       1      5

Performance Metrics:
Accuracy:    91.7%
Precision:   100.0%
Recall:      83.3%
F1-Score:    90.9%

🎯 EXCELLENT: System performance is excellent (≥90% accuracy)
✅ No false alarms - good operational efficiency
⚠️  1 missed fault - review detection threshold
```

## 🔧 Customization Guide

### Adjusting the Anomaly Threshold
```c
// In stm32_test_data.h, modify:
#define ANOMALY_THRESHOLD   819     // Current: 0.05 * 16384

// Lower threshold = more sensitive (fewer missed faults, more false alarms)
#define ANOMALY_THRESHOLD   655     // 0.04 * 16384 - more sensitive

// Higher threshold = less sensitive (more missed faults, fewer false alarms)  
#define ANOMALY_THRESHOLD   983     // 0.06 * 16384 - less sensitive
```

### Adding Your Own Test Cases
```c
// Add to test_cases array in stm32_test_data.h:
{
    .features = {
        2048,   // RMS in Q14
        4096,   // Peak in Q14
        // ... other 6 features
    },
    .expected_label = LABEL_NORMAL,  // or LABEL_FAULT
    .description = "Your test case description",
    .expected_error = 0.020f
}
```

### Scaling Parameters
The test data uses MinMax scaling. Update these parameters to match your trained model:
```c
// In stm32_test_data.h:
static const int32_t scaler_min_fixed[NUM_FEATURES] = {
    819,    // RMS min: 0.05 * 16384
    // ... update with your scaler.data_min_ values
};

static const int32_t scaler_max_fixed[NUM_FEATURES] = {
    13107,  // RMS max: 0.8 * 16384  
    // ... update with your scaler.data_max_ values
};
```

## 🔍 Debugging Tips

### Individual Test Execution
```c
// Test only normal cases
test_normal_bearings();

// Test only fault cases  
test_faulty_bearings();

// Run specific test case
run_single_test(7);  // Test case 8 (developing fault)
```

### Feature Analysis
```c
// Print scaled features for debugging
const test_case_t* test_case = get_test_case(0);
float scaled_features[NUM_FEATURES];
scale_features(test_case->features, scaled_features);

printf("Scaled features: ");
for (int i = 0; i < NUM_FEATURES; i++) {
    printf("%.3f ", scaled_features[i]);
}
printf("\\n");
```

### Model Output Analysis
```c
// Check model outputs vs inputs
printf("Input:  [%.3f, %.3f, %.3f, ...]\\n", input->data.f[0], input->data.f[1], input->data.f[2]);
printf("Output: [%.3f, %.3f, %.3f, ...]\\n", output->data.f[0], output->data.f[1], output->data.f[2]);
```

## 📈 Performance Optimization

### Memory Usage
- **Tensor Arena**: 8KB (adjust TENSOR_ARENA_SIZE if needed)
- **Test Data**: ~2KB static memory
- **Stack Usage**: ~1KB for buffers

### Timing Analysis
Add timing measurements:
```c
uint32_t start_time = HAL_GetTick();
float error = run_inference(scaled_features);
uint32_t inference_time = HAL_GetTick() - start_time;
printf("Inference time: %lu ms\\n", inference_time);
```

### Power Optimization
```c
// Enter low power mode between tests
HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);
```

## 🛠️ Integration Checklist

- [ ] Include `stm32_test_data.h` and `stm32_test_runner.c` in your project
- [ ] Ensure `model_data.h` is generated from your .tflite file
- [ ] Configure UART for debug output (115200 baud recommended)
- [ ] Allocate sufficient stack size (≥4KB recommended)
- [ ] Enable floating-point unit if available
- [ ] Test with different optimization levels (-O2 recommended)
- [ ] Verify memory constraints (Flash: ~50KB, RAM: ~12KB)

## 🔧 Build Configuration

### GCC Compiler Flags
```makefile
CFLAGS += -DARM_MATH_CM4
CFLAGS += -D__FPU_PRESENT=1
CFLAGS += -mfloat-abi=hard -mfpu=fpv4-sp-d16
CFLAGS += -O2 -g
```

### Required Libraries
```makefile
# TensorFlow Lite Micro
LIBS += -ltensorflow-microlite

# ARM CMSIS DSP (optional, for optimized math)
LIBS += -larm_cortexM4lf_math
```

## 📝 Troubleshooting

### Common Issues

**❌ "AllocateTensors() failed"**
- Increase TENSOR_ARENA_SIZE
- Check available RAM

**❌ "Model schema version not supported"**  
- Regenerate model_data.h from latest .tflite file
- Update TensorFlow Lite Micro library

**❌ "Input tensor dimensions incorrect"**
- Verify model expects 8 input features
- Check model architecture matches training

**❌ High false negative rate**
- Lower ANOMALY_THRESHOLD value
- Verify scaler parameters match training

**❌ High false positive rate**
- Raise ANOMALY_THRESHOLD value
- Check for numerical precision issues

### Debug Output Example
Enable verbose debugging by adding `-DDEBUG_VERBOSE` to compile flags.

## 🎯 Deployment Validation

Before deploying to production:
1. Run `validate_system()` - must return HAL_OK
2. Achieve ≥90% accuracy on test suite
3. Zero false negatives (missed faults)
4. ≤1 false positive (false alarm)
5. Inference time <100ms per sample
6. Memory usage within MCU constraints

## 📞 Support

If you encounter issues:
1. Check test output for specific error messages
2. Verify model_data.h matches your trained model
3. Ensure scaler parameters are correctly converted
4. Test with known good hardware configuration
5. Compare results with Python reference implementation

---

**✅ With these test files, you can thoroughly validate your STM32 TinyML bearing fault detection system without needing physical sensors!**