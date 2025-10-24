# 🛰️ Visual Search & Auto-Annotation for Satellite Imagery

**GPU-accelerated AI pipeline for object detection, similarity search, and auto-annotation in 3 m-resolution satellite imagery.**  
Developed under the **SvarAikyam AI / AI Fusion** research initiative, this repository integrates **DINOv2** embeddings, **Segment Anything 2 (SAM2)** polygonization, and classic **LBP + NDWI** features for fully automated candidate detection and review.

---

## 🌍 Overview

This framework enables **rapid auto-labeling of geospatial objects** such as:

> Brick Kiln | STP | Solar Panel | Sheds | Metro Shed | Play Ground | Pond-1 | Pond-2

using **prototype-based matching** between known polygon samples and new satellite tiles.

### 🔹 Core Pipelines
| Stage | Description |
|-------|--------------|
| **Prototype Creation** | `create_features_dino.py` / `create_features_LBP_RGB.py` extract DINOv2 or LBP+NDWI feature vectors from annotated `.json` polygons. |
| **Batch Prototype Builder** | `batch_create_dino_prototypes.sh` automates feature generation for all TIF + JSON pairs. |
| **Auto-Annotation (GPU Optimized)** | `auto_annotate_dino_nvtx_optimized.py` performs window-based candidate detection with FAISS similarity search and optional SAM2 mask verification. |
| **Batch Detection** | `batch_auto_annotate_dino.sh` runs the optimized annotator across the dataset with resume-safe logic. |
| **Interactive Review UI** | `ui_review.py` provides OpenCV-based manual correction, polygonization, and YOLO export. |

---

## ⚙️ GPU Profiling & Optimization

Integrated support for **Nsight Systems / Nsight Compute / Torch Profiler** via  
`auto_annotate_dino_profiled.py`, enabling per-kernel timing, memory profiling, and NVTX range analysis.  
This project also experiments with **Agentic GPU Optimization** tools (see [aifusion.in](https://aifusion.in)) for autotuning CUDA workloads on RTX 3060 / Jetson Orin / Qualcomm RB5.

---

## 🖼️ Example Result

Annotated output from the full DINOv2 + SAM2 pipeline:

<p align="center">
  <img src="results/sat_det.png" alt="Annotated Satellite Detection Result" width="720">
</p>

---

## 🧩 Dependencies

- Python ≥ 3.10  
- PyTorch ≥ 2.2 with CUDA 12.x  
- Transformers (HuggingFace)  
- Rasterio, OpenCV, NumPy, Scikit-image, Pandas  
- [Segment-Anything 2](https://github.com/facebookresearch/segment-anything-2) for SAM2 polygonization  

```bash
pip install torch torchvision torchaudio transformers rasterio opencv-python scikit-image pandas tqdm shapely
