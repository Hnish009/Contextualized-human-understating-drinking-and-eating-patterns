"""
Quick script to check if TensorFlow is using GPU or CPU
""" 

import tensorflow as tf
import sys

print("=" * 60)
print("TensorFlow GPU/CPU Check")
print("=" * 60)

print(f"\nTensorFlow version: {tf.__version__}")

# Check for GPU
print("\nGPU Devices:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  ✓ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"    GPU {i}: {gpu}")
        # Get GPU details
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"      Details: {gpu_details}")
        except:
            pass
else:
    print("  ✗ No GPU devices found")

print("\nCPU Devices:")
cpus = tf.config.list_physical_devices('CPU')
print(f"  ✓ Found {len(cpus)} CPU device(s)")

# Check if GPU is available for computation
print("\nGPU Available for Computation:")
if tf.test.is_gpu_available():
    print("  ✓ Yes - TensorFlow will use GPU")
else:
    print("  ✗ No - TensorFlow will use CPU")

# Show what device will be used
print("\n" + "=" * 60)
print("Default Device:")
print("=" * 60)

# Create a simple tensor to see where it's placed
with tf.device('/CPU:0'):
    cpu_tensor = tf.constant([1.0, 2.0, 3.0])
    print(f"CPU tensor: {cpu_tensor.device}")

if gpus:
    with tf.device('/GPU:0'):
        gpu_tensor = tf.constant([1.0, 2.0, 3.0])
        print(f"GPU tensor: {gpu_tensor.device}")
    print("\n✓ GPU is available! TensorFlow will use GPU for training.")
    print("  Training will be much faster (30-60 min vs 2-4 hours)")
else:
    print("\n✗ GPU not available. TensorFlow will use CPU.")
    print("  Training will take longer (2-4 hours)")

print("\n" + "=" * 60)
print("To check during training:")
print("  - Watch Task Manager > Performance > GPU")
print("  - GPU usage should spike during training")
print("  - Or run: nvidia-smi (if you have NVIDIA GPU)")
print("=" * 60)

