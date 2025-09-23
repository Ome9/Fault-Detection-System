#!/usr/bin/env python3
"""
Test script to verify the NASA IMS Bearing Fault Detection system setup
This script checks if all dependencies are installed and working correctly.
"""

import sys
import importlib

def test_imports():
    """Test all required imports"""
    required_packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'), 
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('pathlib', None),
        ('scipy.signal', 'sig'),
        ('scipy.stats', 'stats'),
        ('scipy.fft', None),
        ('sklearn.preprocessing', None),
        ('sklearn.metrics', None),
        ('tensorflow', 'tf'),
    ]
    
    print("Testing required imports...")
    print("-" * 50)
    
    failed_imports = []
    
    for package_name, alias in required_packages:
        try:
            module = importlib.import_module(package_name)
            if alias:
                globals()[alias] = module
            print(f"✓ {package_name}")
        except ImportError as e:
            print(f"✗ {package_name} - {e}")
            failed_imports.append(package_name)
    
    print("-" * 50)
    
    if failed_imports:
        print(f"\n⚠ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are available!")
        return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    print("-" * 50)
    
    try:
        import numpy as np
        
        # Test numpy
        test_signal = np.random.normal(0, 1, 1024)
        print(f"✓ NumPy: Generated test signal with shape {test_signal.shape}")
        
        # Test scipy if available
        try:
            from scipy.fft import fft
            fft_result = fft(test_signal)
            print(f"✓ SciPy: FFT computed, shape {fft_result.shape}")
        except ImportError:
            print("- SciPy: Not available for FFT test")
        
        # Test sklearn if available  
        try:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(test_signal.reshape(-1, 1))
            print(f"✓ Scikit-learn: StandardScaler working")
        except ImportError:
            print("- Scikit-learn: Not available")
            
        # Test TensorFlow if available
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow: Version {tf.__version__} available")
        except ImportError:
            print("- TensorFlow: Not available (required for training)")
            
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("NASA IMS Bearing Fault Detection - Environment Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    if imports_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"{'✓' if imports_ok else '✗'} Package imports")
    print(f"{'✓' if functionality_ok else '✗'} Basic functionality")
    
    if imports_ok and functionality_ok:
        print("\n🎉 Environment is ready! You can run 'python Code.py'")
    else:
        print("\n⚠ Environment needs setup. Run 'python setup.py' first")
    
    print("=" * 60)

if __name__ == "__main__":
    main()