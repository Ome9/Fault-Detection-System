# Deployment Guide

## Overview

This guide covers deploying the refined bearing fault detection model to STM32 microcontrollers.

## Model Information

- **Performance**: 75.5% F1-score, 88.0% accuracy
- **Model Size**: 15.4 KB (TensorFlow Lite)
- **Features**: 16 optimized bearing indicators
- **Architecture**: Autoencoder (16→48→24→12→24→48→16)
- **Threshold**: 0.045142 (reconstruction error)

## Deployment Files

### Generated Files
- `refined_model.tflite`: TensorFlow Lite model
- `refined_model_data.h`: C header with model weights (15,768 bytes)
- `refined_model_test.c`: Testing and validation framework

### Required Components
- TensorFlow Lite Micro library
- STM32 HAL drivers for ADC and UART
- Math libraries (arm_math.h recommended)

## Hardware Setup

### Minimum Requirements
- **MCU**: STM32F4 series or higher (ARM Cortex-M4)
- **Flash**: 64KB minimum (32KB for application, 32KB for model)
- **RAM**: 32KB minimum (16KB for tensor arena, 16KB for application)
- **Clock**: 84MHz minimum for real-time processing

### Recommended Hardware
- **Board**: STM32F446RE Nucleo board
- **Sensor**: ADXL345 accelerometer or equivalent
- **Sampling**: 20kHz ADC sampling rate
- **Connection**: I2C or SPI for sensor communication

## Software Integration

### 1. Include Model Data

```c
#include "refined_model_data.h"

// Model is available as:
// const unsigned char refined_model_data[15768];
// const unsigned int refined_model_len = 15768;
```

### 2. Initialize TensorFlow Lite

```c
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Tensor arena size (adjust based on available RAM)
constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Initialize TensorFlow Lite
tflite::AllOpsResolver resolver;
const tflite::Model* model = tflite::GetModel(refined_model_data);
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, 
                                    kTensorArenaSize, &error_reporter);
interpreter.AllocateTensors();
```

### 3. Feature Extraction

```c
void extract_features(const float* signal, int length, float* features) {
    // Extract 16 features from vibration signal
    // Features: RMS, Peak, Crest Factor, Kurtosis, Skewness, etc.
    // See refined_model_test.c for complete implementation
}
```

### 4. Inference

```c
float run_inference(const float* features) {
    // Copy features to input tensor
    TfLiteTensor* input = interpreter.input(0);
    for (int i = 0; i < 16; i++) {
        input->data.f[i] = features[i];
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        return -1.0f;  // Error
    }
    
    // Calculate reconstruction error
    TfLiteTensor* output = interpreter.output(0);
    float reconstruction_error = 0.0f;
    for (int i = 0; i < 16; i++) {
        float diff = features[i] - output->data.f[i];
        reconstruction_error += diff * diff;
    }
    
    return reconstruction_error / 16.0f;
}
```

### 5. Anomaly Detection

```c
#define ANOMALY_THRESHOLD 0.045142f

bool detect_fault(const float* signal, int length) {
    float features[16];
    extract_features(signal, length, features);
    
    float reconstruction_error = run_inference(features);
    
    return (reconstruction_error > ANOMALY_THRESHOLD);
}
```

## Performance Optimization

### Memory Optimization
- Use static allocation for all buffers
- Minimize tensor arena size based on model requirements
- Consider quantization if memory is limited

### Processing Optimization
- Use ARM CMSIS-DSP library for feature extraction
- Implement circular buffers for continuous data processing
- Optimize feature extraction with fixed-point arithmetic if needed

### Real-time Considerations
- Process data in 4096-sample windows (205ms at 20kHz)
- Use DMA for ADC data transfer
- Implement double buffering for continuous processing

## Testing and Validation

### Using the Test Framework
```bash
cd tests
make test              # Run automated tests
make interactive       # Manual testing mode
```

### Validation Checklist
- [ ] Model loads correctly
- [ ] Feature extraction matches Python implementation
- [ ] Inference produces expected results
- [ ] Memory usage within limits
- [ ] Real-time performance achieved
- [ ] Fault detection accuracy verified

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Verify model data is correctly included
   - Check tensor arena size
   - Ensure TensorFlow Lite version compatibility

2. **Poor Performance**
   - Validate feature extraction implementation
   - Check sampling rate and signal quality
   - Verify threshold value

3. **Memory Issues**
   - Reduce tensor arena size
   - Use quantized model if available
   - Optimize feature extraction buffer sizes

4. **Real-time Issues**
   - Increase MCU clock frequency
   - Optimize feature extraction algorithms
   - Use hardware accelerators (FPU, DSP)

### Performance Monitoring
- Monitor inference time (should be <10ms)
- Track memory usage during operation
- Validate reconstruction error ranges
- Test with known fault/normal conditions

## Advanced Features

### Continuous Learning
- Implement threshold adaptation based on operating conditions
- Store and analyze long-term trends
- Update model parameters for specific machinery

### Communication
- Send results via UART/CAN for system integration
- Implement wireless transmission (WiFi/LoRa)
- Log data for offline analysis

### Multi-sensor Fusion
- Combine multiple accelerometer readings
- Integrate temperature and acoustic sensors
- Implement sensor validation and redundancy

## References

- TensorFlow Lite Micro documentation
- STM32 HAL documentation
- ARM CMSIS-DSP library reference
- Bearing fault detection research papers