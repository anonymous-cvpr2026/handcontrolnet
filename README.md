# HandControlNet (CVPR 2026 Anonymous Submission)

> Lightweight ControlNet adapter for anime hand anomaly mitigation using only 677 images.

## Files
- `training_data.json`: 677 annotated anime hand images (bbox)
- `advanced_controlnet.pth`: Trained ControlNet weights
- `advanced_controlnet.py`: Training & inference code
- `hand_annotation_tool.zip`: Visualization tool

## Quick Start
```bash
pip install torch diffusers transformers accelerate
python advanced_controlnet.py --infer
