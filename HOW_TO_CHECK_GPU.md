# How to Check if TensorFlow is Using GPU or CPU

## Method 1: Run the Check Script

After installing TensorFlow, run:
```bash
python check_gpu.py
```

This will show:
- GPU devices detected
- Whether TensorFlow can use GPU
- Default device being used

---

## Method 2: Check During Training

The training scripts (`train_model1.py` and `train_model2.py`) now automatically show GPU/CPU status at the start.

Look for messages like:
- `✓ GPU detected: 1 device(s)` → Using GPU
- `✗ No GPU detected - using CPU` → Using CPU

---

## Method 3: Watch Task Manager (Windows)

1. Open **Task Manager** (Ctrl+Shift+Esc)
2. Go to **Performance** tab
3. Look for **GPU** section
4. **During training:**
   - If GPU usage spikes → Using GPU ✓
   - If only CPU usage increases → Using CPU ✗

---

## Method 4: Check NVIDIA GPU (if you have NVIDIA)

Open PowerShell/Command Prompt and run:
```bash
nvidia-smi
```

**If GPU is working:**
- You'll see GPU info, memory usage, processes
- You'll see `python.exe` or `pythonw.exe` in the processes list during training

**If you get "command not found":**
- Either no NVIDIA GPU, or drivers not installed

---

## Method 5: Check in Python Code

Run this in Python:
```python
import tensorflow as tf

# Check devices
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("CPU devices:", tf.config.list_physical_devices('CPU'))

# Check if GPU available
print("GPU available:", tf.test.is_gpu_available())
```

---

## Method 6: Watch Training Speed

**GPU Training (RTX 3060):**
- Model 1: ~30-60 minutes
- Model 2: ~2-5 minutes
- Loss decreases quickly

**CPU Training:**
- Model 1: ~2-4 hours
- Model 2: ~10-20 minutes
- Slower progress

---

## Troubleshooting: GPU Not Detected

If you have an RTX 3060 but GPU isn't detected:

1. **Install CUDA and cuDNN:**
   - TensorFlow needs CUDA/cuDNN for GPU support
   - Check TensorFlow version requirements
   - For TF 2.13: CUDA 11.8, cuDNN 8.6

2. **Install TensorFlow GPU version:**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-gpu
   ```
   Or use conda:
   ```bash
   conda install tensorflow-gpu
   ```

3. **Check GPU drivers:**
   - Make sure NVIDIA drivers are installed
   - Run `nvidia-smi` to verify

4. **Verify GPU in Device Manager:**
   - Open Device Manager
   - Look under "Display adapters"
   - Should see your RTX 3060

---

## Quick Reference

**Signs you're using GPU:**
- ✓ Training is fast (Model 1: <1 hour)
- ✓ Task Manager shows GPU usage
- ✓ nvidia-smi shows Python process
- ✓ Training script says "GPU detected"

**Signs you're using CPU:**
- ✗ Training is slow (Model 1: >2 hours)
- ✗ Only CPU usage in Task Manager
- ✗ Training script says "No GPU detected"
- ✗ nvidia-smi not found or shows no Python process

---

**Note:** CPU training still works perfectly fine! It just takes longer. For demonstration purposes, CPU is perfectly acceptable.

