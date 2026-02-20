from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, preprocess
from PIL import Image
import numpy as np
import os
import gc
import onnx

# Preprocess model
preprocess.quant_pre_process(
    input_model_path='best.onnx',
    output_model_path='best_preprocessed.onnx',
    skip_optimization=False,
)
print("Preprocessing done!")

# Auto-detect ALL sensitive nodes to exclude
model = onnx.load('best_preprocessed.onnx')
nodes_to_exclude = []
sensitive_ops = ['Sigmoid', 'Mul', 'Concat', 'Resize', 'Reshape',
                 'Transpose', 'Split', 'Softmax', 'Add', 'Sub', 'Div']

for node in model.graph.node:
    if node.op_type in sensitive_ops:
        if node.name:
            nodes_to_exclude.append(node.name)

print(f"Excluding {len(nodes_to_exclude)} sensitive nodes (only quantizing Conv layers)")

class YOLOCalibReader(CalibrationDataReader):
    def __init__(self, calib_dir, img_size=320, num_images=100):
        self.img_size = img_size
        self.imgs = [
            os.path.join(calib_dir, f)
            for f in os.listdir(calib_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ][:num_images]
        self.idx = 0
        print(f"Using {len(self.imgs)} calibration images")

    def get_next(self):
        if self.idx >= len(self.imgs):
            return None
        try:
            img = Image.open(self.imgs[self.idx]).convert('RGB')
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
            img.close()
            del img
            gc.collect()
            self.idx += 1
            print(f"Calibrating image {self.idx}/{len(self.imgs)}", end='\r')
            return {'images': arr}
        except Exception as e:
            print(f"\nSkipping {self.imgs[self.idx]}: {e}")
            self.idx += 1
            return self.get_next()

print("Starting quantization...")

quantize_static(
    model_input='best_preprocessed.onnx',
    model_output='best_int8.onnx',
    calibration_data_reader=YOLOCalibReader('calib_images/', num_images=100),
    quant_format=QuantFormat.QDQ,
    per_channel=False,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
    calibrate_method=CalibrationMethod.Entropy,
    nodes_to_exclude=nodes_to_exclude,
    extra_options={
        'ActivationSymmetric': False,
        'WeightSymmetric': True,
        'EnableSubgraph': False,
        'ForceQuantizeNoInputCheck': False,
        'MatMulConstBOnly': True,
        'AddQDQPairToWeight': False,
    }
)

print("\nDone! Quantized model saved as best_int8.onnx")
print(f"Original size:   {os.path.getsize('best.onnx')/1e6:.2f} MB")
print(f"Quantized size:  {os.path.getsize('best_int8.onnx')/1e6:.2f} MB")
