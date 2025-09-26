# Testing Guide

## Overview

This guide covers the comprehensive testing framework for the refined bearing fault detection model. The testing system allows validation without physical sensors through advanced signal simulation.

## Test Framework Architecture

### Core Components
- **refined_model_test.c**: Main C testing implementation
- **generate_test_data.py**: Python test data generator
- **Makefile**: Build system for multiple platforms
- **run_tests.bat**: Windows test runner script

### Testing Modes
1. **Automated Test Suite**: 10 predefined bearing conditions
2. **Interactive Testing**: Custom parameter input
3. **Batch Validation**: Large-scale test vector processing
4. **Performance Benchmarking**: Speed and memory analysis

## Quick Start

### Windows
```batch
cd tests
run_tests.bat
```

### Linux/Mac
```bash
cd tests
make test
```

### Interactive Mode
```bash
make interactive
# Or directly:
./build/refined_model_test interactive
```

## Test Cases

### Built-in Test Scenarios

| Test ID | Description | Type | Expected Result |
|---------|-------------|------|-----------------|
| 1 | Normal bearing - healthy operation | Normal | PASS |
| 2 | Normal bearing - slight load variation | Normal | PASS |
| 3 | Normal bearing - light load | Normal | PASS |
| 4 | Normal bearing - speed variation | Normal | PASS |
| 5 | Inner race fault - 2x harmonics | Fault | FAULT DETECTED |
| 6 | Outer race fault - 3x harmonics | Fault | FAULT DETECTED |
| 7 | Ball fault - 4x harmonics | Fault | FAULT DETECTED |
| 8 | Severe inner race fault | Fault | FAULT DETECTED |
| 9 | Multiple fault conditions | Fault | FAULT DETECTED |
| 10 | Critical bearing failure | Fault | FAULT DETECTED |

### Test Parameters

Each test case includes:
- **Amplitude**: Vibration magnitude (0.5-7.5)
- **Frequency**: Base frequency in Hz (50-420)
- **Noise Level**: Background noise (0.05-0.8)
- **Fault Type**: Normal/Fault classification
- **Description**: Human-readable test description

## Signal Simulation

### Vibration Signal Generation

The test framework generates realistic bearing vibration signals:

```c
void generate_vibration_signal(const bearing_condition_t *condition, 
                              float *signal, int length);
```

**Normal Bearing Characteristics**:
- Single fundamental frequency
- Low harmonic content
- Minimal amplitude modulation
- Gaussian noise

**Faulty Bearing Characteristics**:
- Multiple harmonics (2x, 3x fundamental)
- Amplitude modulation (typical of bearing faults)
- Impulsive components (bearing impacts)
- Increased noise levels

### Feature Extraction Validation

The C implementation extracts 16 features matching the Python model:

1. **RMS** - Root Mean Square value
2. **Peak** - Maximum absolute value
3. **Crest Factor** - Peak/RMS ratio
4. **Kurtosis** - Fourth statistical moment
5. **Skewness** - Third statistical moment
6. **Standard Deviation** - Signal variability
7. **Mean Absolute** - Average absolute value
8. **Peak-to-Peak** - Maximum range
9. **Clearance Factor** - Peak/(Mean Absolute)²
10. **Shape Factor** - RMS/Mean Absolute
11. **Impulse Factor** - Peak/Mean Absolute
12. **Envelope RMS** - Envelope analysis
13. **Spectral Energy** - Frequency domain energy
14. **Mean** - Average value
15. **Median** - Middle value
16. **Percentile Range** - 90th - 10th percentile

## Test Data Generation

### Python Test Generator

Generate additional test vectors:

```bash
python generate_test_data.py
```

**Generated Files**:
- `comprehensive_test_vectors.json`: Test data in JSON format
- `c_test_data.h`: C header with test arrays
- `test_plots/`: Visualization plots

### Custom Test Data

Create custom test conditions:

```python
generator = TestDataGenerator()
test_vectors = generator.generate_test_vectors(
    num_normal=50,
    num_faulty=50
)
```

## Performance Validation

### Expected Performance
- **Test Accuracy**: >80% (based on 75.5% F1-score model)
- **Processing Time**: <1ms per inference
- **Memory Usage**: <16KB RAM
- **False Positives**: <8% (normal conditions)
- **False Negatives**: <26% (fault conditions)

### Benchmark Results

Run performance benchmark:

```bash
make benchmark
```

**Typical Results**:
- Inference time: 0.1-0.5ms
- Feature extraction: 0.2-0.8ms
- Total processing: <1.5ms per sample
- Memory usage: 8-12KB

## Interactive Testing

### Manual Test Mode

Enter interactive mode for custom testing:

```
Enter parameters: 2.5 120 0.3
   Reconstruction Error: 0.123456
   Prediction: FAULT DETECTED
   Confidence: 87.3%

Enter parameters: 1.0 60 0.1
   Reconstruction Error: 0.023456
   Prediction: NORMAL OPERATION
   Confidence: 91.2%
```

### Parameter Guidelines

**Normal Conditions**:
- Amplitude: 0.5-1.5
- Frequency: 50-80 Hz
- Noise: 0.05-0.2

**Fault Conditions**:
- Amplitude: 2.0-7.0
- Frequency: 100-400 Hz
- Noise: 0.3-0.8

## Validation Against Python Model

### Consistency Checks

The test framework validates that the C implementation produces results consistent with the Python model:

1. **Feature Extraction**: C features match Python within 1%
2. **Model Inference**: Reconstruction errors within 5%
3. **Anomaly Detection**: Same classification decisions
4. **Performance**: Similar accuracy on test data

### Validation Process

```bash
python generate_test_data.py    # Generate reference data
make test                       # Run C implementation
# Compare results automatically
```

## Troubleshooting

### Common Issues

1. **Compilation Errors**
   ```bash
   make clean
   make check-model    # Verify model files exist
   make install-deps   # Install build tools
   ```

2. **Model Loading Issues**
   - Verify `refined_model_data.h` exists in `../refined_deployment/`
   - Check file permissions and path
   - Ensure model file is not corrupted

3. **Poor Test Performance**
   - Validate feature extraction implementation
   - Check threshold value (0.045142)
   - Compare with Python reference implementation

4. **Memory Issues**
   ```bash
   make memcheck       # Run Valgrind if available
   ```

### Debug Mode

Enable verbose output:
```c
#define DEBUG_MODE 1    // In refined_model_test.c
```

## Advanced Testing

### Stress Testing

Generate large test datasets:
```python
# In generate_test_data.py
test_vectors = generator.generate_test_vectors(
    num_normal=1000,
    num_faulty=1000
)
```

### Cross-Platform Testing

Test on different platforms:
```bash
# Linux
make CC=gcc test

# Windows MinGW
make CC=x86_64-w64-mingw32-gcc test

# ARM Cross-compilation
make CC=arm-none-eabi-gcc test
```

### Hardware-in-Loop Testing

For systems with actual sensors:
1. Replace signal generation with ADC reading
2. Validate real-time performance
3. Compare with known bearing conditions
4. Calibrate threshold for specific hardware

## Test Reports

### Automated Reporting

The test framework generates detailed reports:
```
📈 TEST RESULTS SUMMARY:
   Total Tests: 10
   Correct Predictions: 9
   Test Accuracy: 90.0%
   Model Threshold: 0.045142

🎉 TEST PASSED: Model performance meets expectations!
```

### Custom Analysis

Generate visualizations:
```python
generator.visualize_test_data(test_vectors)
```

**Generated Plots**:
- Feature distributions (normal vs fault)
- Correlation matrix
- Performance metrics
- ROC curves

## Integration Testing

### STM32 Integration

Test with actual STM32 hardware:
1. Use test framework as reference
2. Validate feature extraction on hardware
3. Compare inference results
4. Verify real-time performance

### System Integration

Test within larger systems:
- Communication protocols
- Data logging
- Alert mechanisms
- User interfaces

## Continuous Testing

### Automated CI/CD

Integrate with version control:
```yaml
# GitHub Actions example
- name: Run Model Tests
  run: |
    cd tests
    make test
```

### Regression Testing

Ensure model changes don't break functionality:
1. Baseline test results
2. Compare after model updates
3. Validate performance metrics
4. Check for breaking changes