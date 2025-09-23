# NASA IMS Bearing Fault Detection System

## 🔍 Overview

This project implements an advanced **Autoencoder-based Fault Detection System** for bearing analysis using the NASA IMS (Intelligent Maintenance Systems) Bearing Dataset. The system is specifically designed and optimized for deployment on **STM32 F446RE** microcontrollers, enabling real-time bearing condition monitoring in industrial environments.

## 🎯 Key Features

- **Advanced Feature Extraction**: 16 optimized features including time-domain, frequency-domain, and wavelet-based features
- **Deep Learning Model**: Autoencoder neural network for anomaly detection
- **Real NASA Data**: Trained on authentic NASA IMS bearing run-to-failure datasets
- **STM32 Ready**: Optimized C code generation for STM32 F446RE microcontroller
- **Multiple Test Sets**: Supports all 3 NASA IMS test sets with different failure modes
- **Comprehensive Analysis**: Includes visualization, performance metrics, and risk assessment

## 📊 Dataset Information

The NASA IMS Bearing Dataset consists of:
- **Test Set 1**: 2,156 files, failures in bearing 3 (inner race) and bearing 4 (roller)
- **Test Set 2**: 984 files, outer race failure in bearing 1  
- **Test Set 3**: 4,448 files, outer race failure in bearing 3
- **Sampling Rate**: 20kHz, 20,480 points per file (1-second snapshots)
- **Bearings**: Rexnord ZA-2115 double row bearings
- **Operating Conditions**: 2000 RPM, 6000 lbs radial load

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy pathlib
```

### Usage

1. **Download the NASA IMS Dataset**:
   - Visit: [NASA Prognostics Center of Excellence Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
   - Download "IMS Rexnord Bearing Data.zip"
   - Extract to your desired location

2. **Update Dataset Path**:
   ```python
   # In Code.py, line ~595
   dataset_path = "path/to/your/unzipped/IMS/folder"
   ```

3. **Run the System**:
   ```bash
   python Code.py
   ```

### Using Synthetic Data (Demo)

If you don't have the NASA dataset, the system will automatically generate synthetic bearing data for demonstration:

```python
python Code.py  # Will detect missing dataset and use synthetic data
```

## 🏗️ Project Structure

```
errorDetection/
├── Code.py                              # Main implementation
├── convert_model.py                     # Model conversion utilities  
├── stm32_nasa_bearing_detection.c      # Generated STM32 C code
├── nasa_bearing_autoencoder.keras      # Trained Keras model
├── scaler.pkl                          # Feature scaling parameters
├── threshold.npy                       # Anomaly detection threshold
├── 1st_test/                           # NASA Test Set 1 data
├── 2nd_test/                           # NASA Test Set 2 data  
├── 3rd_test/                           # NASA Test Set 3 data
└── .venv/                              # Virtual environment
```

## 🧠 System Architecture

### Feature Extraction Pipeline

1. **Time-Domain Features**:
   - RMS (Root Mean Square)
   - Peak value and Crest Factor
   - Statistical moments (Skewness, Kurtosis)
   - Shape and Impulse factors

2. **Frequency-Domain Features**:
   - Spectral centroid and spread
   - Bearing-specific frequency bands
   - Dominant frequency analysis
   - Power distribution across fault frequencies

3. **Wavelet Features**:
   - Energy ratio between frequency bands
   - High/low frequency energy distribution

### Deep Learning Model

- **Architecture**: Deep Autoencoder
- **Input Layer**: 16 features
- **Encoder**: 32 → 16 → 8 neurons
- **Decoder**: 8 → 16 → 32 → 16 neurons
- **Activation**: ReLU (hidden), Linear (output)
- **Regularization**: L2, Batch Normalization, Dropout

### Anomaly Detection

- **Method**: Reconstruction error threshold
- **Threshold**: 99th percentile of normal data errors
- **Risk Levels**: LOW, MEDIUM, HIGH, CRITICAL
- **Performance**: Optimized for bearing fault patterns

## 📈 Performance Metrics

The system provides comprehensive evaluation including:
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Anomaly detection quality  
- **F1-Score**: Balanced performance metric
- **ROC Analysis**: Threshold sensitivity analysis
- **Condition-Specific**: Performance by bearing condition

## 🔧 STM32 Deployment

### Hardware Requirements
- **MCU**: STM32 F446RE (or compatible)
- **Memory**: ~32KB Flash, ~8KB RAM
- **Peripherals**: ADC, UART, Timer
- **Sensors**: Vibration sensor (accelerometer)

### Generated Code Features
- **Real-time Processing**: 1kHz sampling rate
- **Optimized Math**: Fast sqrt, tanh implementations  
- **Memory Efficient**: Fixed-point arithmetic
- **UART Logging**: Results transmission
- **LED Indicators**: Visual fault indication

### Deployment Steps
1. Run `python Code.py` to generate `stm32_nasa_bearing_detection.c`
2. Insert actual trained model weights in the C file
3. Configure hardware peripherals for your board
4. Compile and flash to STM32 F446RE

## 📊 Visualization

The system generates comprehensive plots:
- Reconstruction error distributions by condition
- Confusion matrices for performance assessment
- ROC curves for threshold analysis  
- Timeline views of bearing degradation

## 🔬 Research Applications

This system is valuable for:
- **Predictive Maintenance**: Early fault detection
- **Industrial IoT**: Edge-based monitoring
- **Research**: Bearing fault analysis methodologies
- **Education**: ML applications in mechanical engineering

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature extraction methods
- Alternative neural network architectures
- Support for other microcontrollers
- Enhanced visualization capabilities
- Real-time data acquisition interfaces

## 📝 Technical Details

### Feature Selection Rationale
The 16 selected features are specifically chosen for bearing fault detection based on mechanical engineering principles:
- **Time-domain features** capture overall vibration levels and impulsiveness
- **Frequency-domain features** detect fault-specific frequency signatures
- **Wavelet features** capture transient impacts characteristic of bearing faults

### Model Architecture Justification  
- **Autoencoder approach** is ideal for anomaly detection with limited fault data
- **Bottleneck encoding** forces learning of essential bearing health patterns
- **Reconstruction error** naturally increases with bearing degradation

## 📚 References

1. NASA Ames Prognostics Data Repository
2. Bearing fault detection methodologies
3. Autoencoder-based anomaly detection
4. STM32 microcontroller programming

## 📄 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This system is designed for research and educational purposes. For critical industrial applications, additional validation and safety measures should be implemented.

---
**Author**: AI Assistant  
**Date**: 2025  
**Version**: 1.0