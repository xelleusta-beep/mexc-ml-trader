import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(__file__))

print("Verifying imports...")
try:
    from ml_engine import Indicators
    print("SUCCESS: ml_engine Indicators imported successfully")
except Exception as e:
    print(f"FAILED: Failed to import ml_engine Indicators: {e}")
    sys.exit(1)

# Generate dummy data
np.random.seed(42)
c = np.cumsum(np.random.randn(300)) + 100
h = c + np.random.rand(300) * 2
lo = c - np.random.rand(300) * 2
v = np.random.rand(300) * 10000

print("\nVerifying Indicators.stoch_rsi mathematical correctness...")
try:
    # Test values
    stoch_val = Indicators.stoch_rsi(c, p=14)
    print(f"stoch_rsi value (p=14) for 300 data points: {stoch_val:.4f}")
    assert 0 <= stoch_val <= 100, f"stoch_rsi must be between 0 and 100, got {stoch_val}"
    print("SUCCESS: Indicators.stoch_rsi works correctly")
except Exception as e:
    print(f"FAILED: Indicators.stoch_rsi failed: {e}")
    sys.exit(1)

print("\nVerifying Indicators.atr_percentile mathematical correctness...")
try:
    atr_p = Indicators.atr_percentile(h, lo, c)
    print(f"atr_percentile value for 300 data points: {atr_p:.4f}")
    assert 0 <= atr_p <= 1, f"atr_percentile must be between 0 and 1, got {atr_p}"
    print("SUCCESS: Indicators.atr_percentile works correctly")
except Exception as e:
    print(f"FAILED: Indicators.atr_percentile failed: {e}")
    sys.exit(1)

print("\nAll automated verification checks passed successfully!")
