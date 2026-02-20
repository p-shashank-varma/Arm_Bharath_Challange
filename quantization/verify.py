import onnxruntime as ort
import numpy as np
import os

print("Loading models...")
orig_sess  = ort.InferenceSession('best.onnx',      providers=['CPUExecutionProvider'])
quant_sess = ort.InferenceSession('best_int8.onnx', providers=['CPUExecutionProvider'])

abs_diffs = []
rel_diffs = []

for i in range(5):
    dummy     = np.random.rand(1, 3, 320, 320).astype(np.float32)
    orig_out  = orig_sess.run(None,  {'images': dummy})[0]
    quant_out = quant_sess.run(None, {'images': dummy})[0]

    abs_diff = np.abs(orig_out - quant_out)
    # relative diff — ignore near-zero values to avoid division noise
    mask     = np.abs(orig_out) > 1.0
    rel_diff = (abs_diff[mask] / np.abs(orig_out[mask])).mean() * 100 if mask.any() else 0

    abs_diffs.append(abs_diff.mean())
    rel_diffs.append(rel_diff)

print(f"\n=== Absolute Difference ===")
print(f"Mean : {np.mean(abs_diffs):.4f}")
print(f"Max  : {np.max(abs_diffs):.4f}")
print(f"Min  : {np.min(abs_diffs):.4f}")

print(f"\n=== Relative Difference (what actually matters) ===")
print(f"Mean : {np.mean(rel_diffs):.2f}%")
print(f"Max  : {np.max(rel_diffs):.2f}%")

print(f"\n=== Model Sizes ===")
orig_size  = os.path.getsize('best.onnx') / 1e6
quant_size = os.path.getsize('best_int8.onnx') / 1e6
print(f"Original  : {orig_size:.2f} MB")
print(f"Quantized : {quant_size:.2f} MB")
print(f"Reduction : {(1 - quant_size/orig_size)*100:.1f}%")

print(f"\n=== Verdict ===")
rel = np.mean(rel_diffs)
if rel < 1.0:
    print(f"✅ EXCELLENT — {rel:.2f}% relative error, safe to deploy")
elif rel < 3.0:
    print(f"✅ GOOD — {rel:.2f}% relative error, acceptable for deployment")
elif rel < 7.0:
    print(f"⚠️  MODERATE — {rel:.2f}% relative error, run visual test")
else:
    print(f"❌ POOR — {rel:.2f}% relative error, not recommended")
