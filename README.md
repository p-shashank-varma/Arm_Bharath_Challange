# ARM Bharat SoC PS4 Challenge  
## Fire, Smoke, and Human Detection on AMD/Xilinx Zynq (PYNQ-Z2)

This repository contains an end-to-end embedded AI workflow for the **ARM Bharat SoC PS4 challenge**:
- Train a compact detector in the cloud
- Export to ONNX
- Quantize to INT8
- Compile for a Zynq DPU target
- Prepare for deployment on a Zynq-7000 based PYNQ-Z2 platform

---

## 1) Project Objective (What, Why, Where, How)

### What
Build a practical object detection pipeline for:
- **Fire**
- **Smoke**
- **Human**

### Why
Edge-first inference is required in safety/monitoring workloads where:
- low latency is critical,
- bandwidth/cloud dependency must be reduced,
- and power-efficient deployment is preferred.

### Where
Target hardware: **PYNQ-Z2** (Zynq-7000 SoC), typically used in:
- labs and student embedded AI projects,
- smart surveillance prototypes,
- industrial/environmental safety PoCs.

### How
The implementation uses:
1. **YOLOv8n** training (`training/training_colab.py`)
2. ONNX export (`training/export_to_onnx.py`)
3. Static post-training quantization (`quantization/quantize.py`)
4. Quantized model verification (`quantization/verify.py`)
5. Vitis-AI compile flow (`quantization/compile.py`)

---

## 2) Hardware Platform (Detailed)

### Target Board: PYNQ-Z2 (Zynq-7000)

| Item | Specification |
|---|---|
| SoC | **XC7Z020-1CLG400C** |
| CPU (PS) | Dual-core Arm Cortex-A9 (Processing System) |
| FPGA Fabric (PL) | Zynq-7000 Programmable Logic |
| External RAM | **512 MB DDR3** |
| Non-volatile storage | **16 MB Quad-SPI Flash** |
| On-chip block memory | BRAM blocks are **36 Kib primitives** (commonly treated as ~32 Kib data + parity per block) |
| Typical boot/media | microSD + QSPI boot options |

### Hardware-specific design implications
- **Input resolution set to 320x320** in training/export to reduce memory pressure and improve FPS feasibility on edge hardware.
- Quantization to **INT8** is used to reduce model size and memory bandwidth demand.
- DPU compile path in this repo targets:
  - `/opt/vitis_ai/compiler/arch/DPUCZDX8G/PYNQ-Z2/arch.json`

---

## 3) Software Stack (Detailed)

| Layer | Tools used in this repo |
|---|---|
| Model training | `ultralytics` (YOLOv8) |
| Export | PyTorch/Ultralytics -> ONNX (`opset=12`) |
| Quantization | `onnxruntime.quantization`, `onnx`, `numpy`, `Pillow` |
| DPU compile | Vitis-AI compiler (`vai_c_onnx`) |
| Runtime-side dependencies | `pynq-dpu`, `numpy`, `pillow` |
| Dev environments | Google Colab (training), Ubuntu/Docker (Vitis-AI flow) |

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4) Repository Architecture

```text
.
├── training/
│   ├── training_colab.py      # YOLOv8n training (imgsz=320, epochs=50)
│   ├── export_to_onnx.py      # Export best.pt -> ONNX (opset=12)
│   ├── detection.py           # Inference script for validation images
│   └── README.md              # Dataset notes
├── models/
│   ├── best.pt                # Trained FP model
│   ├── best.onnx              # ONNX exported model
│   └── quantized_best.onnx    # Quantized model artifact
├── quantization/
│   ├── quantize.py            # Static PTQ flow
│   ├── verify.py              # Output drift + size comparison
│   ├── compile.py             # Vitis-AI compilation command
│   └── docker_setup.md        # Docker setup note
└── results/
    └── output_images/         # Sample inference outputs
```

---

## 5) Model Details

### Base model
- **Model family:** Ultralytics YOLOv8
- **Variant used:** `yolov8n.pt` (nano), selected for embedded constraints

### Training configuration used in code
From `training/training_colab.py`:
- `epochs=50`
- `imgsz=320`
- `device=0` (GPU in Colab)
- dataset pointer expected via `dataset.location`

### Classes
The repository references the RoboFlow dataset:
- **Fire**
- **Smoke**
- **Human**

---

## 6) Dataset

Dataset source referenced by repository docs:
- **RoboFlow: “Fire Smoke and Human Detector”**

Workflow:
1. Create/download dataset export in YOLO/COCO-compatible format
2. Place under your training workspace
3. Point `data=` in training script to your dataset YAML

> Note: exact split sizes are not hardcoded in the repository; dataset composition depends on the user’s exported version.

---

## 7) Quantization Process (Hardware-specific)

Implemented in `quantization/quantize.py`:

1. **Model pre-processing**
   - `quant_pre_process(...)` on `best.onnx`
2. **Sensitive node exclusion**
   - excludes selected ops (e.g., `Sigmoid`, `Concat`, `Resize`, `Softmax`, etc.) to preserve detection quality
3. **Calibration reader**
   - custom `CalibrationDataReader`
   - uses up to **100 calibration images**
   - resizes to **320x320**
   - normalizes to `[0, 1]`
4. **Static PTQ settings**
   - `quant_format=QDQ`
   - `weight_type=QInt8`
   - `activation_type=QUInt8`
   - `calibrate_method=Entropy`

Why this matters on Zynq:
- lower model footprint for limited memory,
- lower bandwidth and compute load,
- better fit for DPU integer inference path.

---

## 8) Compilation Flow for PYNQ-Z2 DPU

`quantization/compile.py` captures the command pattern:

```bash
vai_c_onnx \
  -m best_int8.onnx \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/PYNQ-Z2/arch.json \
  -o ./compiled_model \
  -n smoke_fire_human \
  --options '{"input_shape": "1,3,320,320"}'
```

Key point: **`arch.json` must match the deployed DPU architecture**.

---

## 9) Results and Observations

### Available artifacts in this repository
- `models/best.pt` -> **6.0 MB**
- `models/best.onnx` -> **12 MB**
- `models/quantized_best.onnx` -> **3.1 MB**

Approximate size reduction (ONNX -> quantized ONNX):
- `(12 - 3.1) / 12 ~= 74%` reduction

Qualitative outputs:
- `results/output_images/output_image1.png` to `output_image5.png`
- Demonstrate end-to-end detection output generation.

`quantization/verify.py` additionally provides:
- average absolute/relative output drift between FP and quantized models
- automatic quality verdict bands (EXCELLENT/GOOD/MODERATE/POOR)

---

## 10) Deployment-Oriented Runbook

1. Train:
```bash
python training/training_colab.py
```
2. Export:
```bash
python training/export_to_onnx.py
```
3. Quantize:
```bash
python quantization/quantize.py
```
4. Verify quantization:
```bash
python quantization/verify.py
```
5. Compile for DPU in Vitis-AI container using the compile command above.
6. Move compiled artifact to PYNQ-Z2 runtime environment with `pynq-dpu`.

---

## 11) Applications and Engineering Rationale

### Primary applications
- Early fire/smoke warning in indoor/outdoor cameras
- Smart campus/industrial safety monitoring
- Human-presence-aware hazard systems

### Why this architecture is suitable
- YOLOv8n gives a practical accuracy/latency tradeoff for edge.
- INT8 PTQ significantly reduces memory footprint.
- Zynq PS+PL split enables software flexibility + hardware acceleration path.

---

## 12) Limitations and Next Improvements

- Training/inference scripts currently require manual path updates (`dataset.location`, model paths).
- No pinned benchmark table (FPS/mAP on final board) is committed yet.
- `quantization/compile.py` is a command snippet and should be converted to a valid shell script for direct execution.

Recommended next steps:
- add reproducible benchmark protocol (FPS, latency, mAP50, mAP50-95),
- add board-side runtime script with timing logs,
- add CI checks for script syntax and docs consistency.

---

## 13) External Sources / References

1. **PYNQ-Z2 board overview (Xilinx PYNQ docs)**  
   https://github.com/Xilinx/PYNQ/blob/master/docs/source/pynq_overlays/pynqz2.rst
2. **Vitis-AI model development workflow (official docs)**  
   https://github.com/Xilinx/Vitis-AI/blob/master/docsrc/source/docs/workflow-model-development.rst
3. **Vitis-AI ONNX quantizer notes**  
   https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_onnx/README.md
4. **Ultralytics YOLOv8 model documentation**  
   https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md
5. **ONNX Runtime quantization tooling docs**  
   https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/README.md
6. **Dataset platform used by this project**  
   https://roboflow.com/

---
