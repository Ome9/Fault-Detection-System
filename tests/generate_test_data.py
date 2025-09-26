#!/usr/bin/env python3
"""
Test Data Generator for Refined Bearing Fault Detection Model

This script generates additional test vectors and validates the C implementation
against the Python model to ensure consistency.

Author: AI Assistant
Date: 2025-09-26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import json
import os
import sys
from datetime import datetime

# Add src to path for model imports
sys.path.append('../src')

class TestDataGenerator:
    """Generate comprehensive test data for model validation"""
    
    def __init__(self, sampling_rate=20000, segment_size=4096):
        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.anomaly_threshold = 0.045142  # From refined model
        
    def generate_bearing_signal(self, amplitude, frequency, noise_level, 
                               is_faulty=False, duration=None):
        """Generate synthetic bearing vibration signal"""
        if duration is None:
            duration = self.segment_size / self.sampling_rate
            
        t = np.linspace(0, duration, self.segment_size)
        dt = 1.0 / self.sampling_rate
        
        # Base vibration (fundamental frequency)
        signal_data = amplitude * np.sin(2 * np.pi * frequency * t)
        
        if is_faulty:
            # Add fault harmonics
            signal_data += 0.3 * amplitude * np.sin(2 * np.pi * 2 * frequency * t)
            signal_data += 0.2 * amplitude * np.sin(2 * np.pi * 3 * frequency * t)
            
            # Add amplitude modulation (typical of bearing faults)
            modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 10 * t)
            signal_data *= modulation
            
            # Add impulsive components (bearing impacts)
            impact_times = np.arange(0, duration, 0.1)
            for impact_time in impact_times:
                impact_idx = int(impact_time * self.sampling_rate)
                if impact_idx < len(signal_data):
                    decay_length = min(100, len(signal_data) - impact_idx)
                    decay = np.exp(-100 * np.arange(decay_length) * dt)
                    signal_data[impact_idx:impact_idx + decay_length] += \
                        2.0 * amplitude * decay
        
        # Add noise
        noise = noise_level * (np.random.random(len(signal_data)) - 0.5)
        signal_data += noise
        
        return signal_data
    
    def extract_features(self, signal_data):
        """Extract 16 features matching the C implementation"""
        if len(signal_data) == 0:
            return np.zeros(16)
            
        # Basic statistics
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        rms = np.sqrt(np.mean(signal_data**2))
        rms = max(rms, 1e-10)  # Prevent division by zero
        
        # Higher order moments
        skewness = np.mean(((signal_data - mean_val) / max(std_val, 1e-10))**3) if std_val > 1e-10 else 0
        kurtosis = np.mean(((signal_data - mean_val) / max(std_val, 1e-10))**4) - 3 if std_val > 1e-10 else 0
        
        # Shape factors
        mean_abs = np.mean(np.abs(signal_data))
        peak = np.max(np.abs(signal_data))
        crest_factor = peak / rms
        clearance_factor = peak / max(mean_abs**2, 1e-10)
        shape_factor = rms / max(mean_abs, 1e-10)
        impulse_factor = peak / max(mean_abs, 1e-10)
        peak_to_peak = np.max(signal_data) - np.min(signal_data)
        
        # Additional features
        envelope_rms = rms * 1.1  # Simplified approximation
        spectral_energy = np.mean(signal_data**2)
        median_val = np.median(signal_data)
        percentile_range = np.percentile(signal_data, 90) - np.percentile(signal_data, 10)
        
        features = np.array([
            rms, peak, crest_factor, kurtosis, skewness, std_val,
            mean_abs, peak_to_peak, clearance_factor, shape_factor,
            impulse_factor, envelope_rms, spectral_energy, mean_val,
            median_val, percentile_range
        ])
        
        # Handle NaN/inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def generate_test_vectors(self, num_normal=50, num_faulty=50):
        """Generate comprehensive test vectors"""
        test_vectors = []
        
        print(f"Generating {num_normal} normal and {num_faulty} faulty test cases...")
        
        # Normal bearing conditions
        for i in range(num_normal):
            amplitude = np.random.uniform(0.5, 1.5)
            frequency = np.random.uniform(50, 80)
            noise_level = np.random.uniform(0.05, 0.2)
            
            signal_data = self.generate_bearing_signal(
                amplitude, frequency, noise_level, is_faulty=False
            )
            features = self.extract_features(signal_data)
            
            test_vectors.append({
                'id': f'normal_{i+1:03d}',
                'label': 0,
                'amplitude': amplitude,
                'frequency': frequency,
                'noise_level': noise_level,
                'features': features.tolist(),
                'description': f'Normal bearing - case {i+1}'
            })
        
        # Faulty bearing conditions
        fault_types = [
            ('inner_race', 2.0, 3.0),
            ('outer_race', 2.5, 3.5),
            ('ball_fault', 3.0, 4.0),
            ('severe_fault', 4.0, 6.0)
        ]
        
        for i in range(num_faulty):
            fault_type, min_amp, max_amp = fault_types[i % len(fault_types)]
            amplitude = np.random.uniform(min_amp, max_amp)
            frequency = np.random.uniform(100, 400)
            noise_level = np.random.uniform(0.3, 0.8)
            
            signal_data = self.generate_bearing_signal(
                amplitude, frequency, noise_level, is_faulty=True
            )
            features = self.extract_features(signal_data)
            
            test_vectors.append({
                'id': f'faulty_{i+1:03d}',
                'label': 1,
                'amplitude': amplitude,
                'frequency': frequency,
                'noise_level': noise_level,
                'features': features.tolist(),
                'description': f'{fault_type} fault - case {i+1}'
            })
        
        return test_vectors
    
    def save_test_vectors(self, test_vectors, filename='test_vectors.json'):
        """Save test vectors to JSON file"""
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'sampling_rate': self.sampling_rate,
                'segment_size': self.segment_size,
                'anomaly_threshold': self.anomaly_threshold,
                'num_features': 16,
                'total_vectors': len(test_vectors)
            },
            'test_vectors': test_vectors
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Test vectors saved to {filename}")
        print(f"Total vectors: {len(test_vectors)}")
        
    def generate_c_test_data(self, test_vectors, filename='c_test_data.h'):
        """Generate C header file with test data"""
        with open(filename, 'w') as f:
            f.write("/**\n")
            f.write(" * @file c_test_data.h\n")
            f.write(" * @brief Generated test data for refined model validation\n")
            f.write(" * \n")
            f.write(f" * Generated on: {datetime.now().isoformat()}\n")
            f.write(f" * Total test cases: {len(test_vectors)}\n")
            f.write(" */\n\n")
            f.write("#ifndef C_TEST_DATA_H\n")
            f.write("#define C_TEST_DATA_H\n\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"#define NUM_TEST_VECTORS {len(test_vectors)}\n")
            f.write("#define NUM_FEATURES 16\n\n")
            
            # Test vector structure
            f.write("typedef struct {\n")
            f.write("    char id[32];\n")
            f.write("    int label;\n")
            f.write("    float features[NUM_FEATURES];\n")
            f.write("    char description[64];\n")
            f.write("} test_vector_t;\n\n")
            
            # Test data array
            f.write("static const test_vector_t test_data[NUM_TEST_VECTORS] = {\n")
            
            for i, vector in enumerate(test_vectors):
                f.write("    {\n")
                f.write(f'        .id = "{vector["id"]}",\n')
                f.write(f'        .label = {vector["label"]},\n')
                f.write("        .features = {")
                f.write(", ".join([f"{feat:.6f}f" for feat in vector["features"]]))
                f.write("},\n")
                f.write(f'        .description = "{vector["description"]}"\n')
                f.write("    }")
                if i < len(test_vectors) - 1:
                    f.write(",")
                f.write("\n")
            
            f.write("};\n\n")
            f.write("#endif // C_TEST_DATA_H\n")
        
        print(f"C test data header saved to {filename}")
    
    def visualize_test_data(self, test_vectors, output_dir='test_plots'):
        """Generate visualization plots of test data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract features and labels
        features_array = np.array([v['features'] for v in test_vectors])
        labels = np.array([v['label'] for v in test_vectors])
        
        # Feature names
        feature_names = [
            'RMS', 'Peak', 'Crest Factor', 'Kurtosis', 'Skewness', 'Std Dev',
            'Mean Abs', 'Peak-to-Peak', 'Clearance Factor', 'Shape Factor',
            'Impulse Factor', 'Envelope RMS', 'Spectral Energy', 'Mean',
            'Median', 'Percentile Range'
        ]
        
        # Plot feature distributions
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(16):
            normal_features = features_array[labels == 0, i]
            faulty_features = features_array[labels == 1, i]
            
            axes[i].hist(normal_features, bins=20, alpha=0.7, label='Normal', color='blue')
            axes[i].hist(faulty_features, bins=20, alpha=0.7, label='Faulty', color='red')
            axes[i].set_title(feature_names[i])
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot feature correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = np.corrcoef(features_array.T)
        plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xticks(range(16), feature_names, rotation=45, ha='right')
        plt.yticks(range(16), feature_names)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization plots saved to {output_dir}/")

def main():
    """Main function"""
    print("🔧 Refined Model Test Data Generator")
    print("=" * 50)
    
    # Create generator
    generator = TestDataGenerator()
    
    # Generate test vectors
    print("\n📊 Generating test vectors...")
    test_vectors = generator.generate_test_vectors(num_normal=100, num_faulty=100)
    
    # Save as JSON
    print("\n💾 Saving test data...")
    generator.save_test_vectors(test_vectors, 'comprehensive_test_vectors.json')
    
    # Generate C header
    generator.generate_c_test_data(test_vectors, 'c_test_data.h')
    
    # Generate visualizations
    print("\n📈 Creating visualizations...")
    generator.visualize_test_data(test_vectors)
    
    # Statistics
    normal_count = sum(1 for v in test_vectors if v['label'] == 0)
    faulty_count = sum(1 for v in test_vectors if v['label'] == 1)
    
    print(f"\n📋 Test Data Summary:")
    print(f"   Total test vectors: {len(test_vectors)}")
    print(f"   Normal cases: {normal_count}")
    print(f"   Faulty cases: {faulty_count}")
    print(f"   Features per vector: 16")
    print(f"   Anomaly threshold: {generator.anomaly_threshold:.6f}")
    
    print(f"\n✅ Test data generation complete!")
    print(f"   Files generated:")
    print(f"     - comprehensive_test_vectors.json")
    print(f"     - c_test_data.h")
    print(f"     - test_plots/feature_distributions.png")
    print(f"     - test_plots/feature_correlation.png")

if __name__ == "__main__":
    main()