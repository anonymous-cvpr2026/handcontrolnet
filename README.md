# HandControlNet (CVPR 2026 Anonymous Submission)

> Lightweight ControlNet adapter for anime hand anomaly mitigation.

## Files
- `training_data.json`: 677 annotated anime hand images (bbox)
- `advanced_controlnet.py`: Training & inference code
- `hand_annotation_tool.zip`: Annotation visualization tool
- **Model weights**: [anonymous-cvpr2026/handcontrolnet](https://huggingface.co/anonymous-cvpr2026/handcontrolnet)

## Quick Inference
```bash
pip install -r requirements.txt
# Download model from HF, then:
python advanced_controlnet.py --infer
