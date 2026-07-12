# Graph-Based Framework for Activity Recognition in Egocentric Videos

## Abstract
We propose a graph-based framework for activity recognition in egocentric videos, representing each frame as an **Action Scene Graph (ASG)** constructed from a VLM caption. We introduce two graph variants:

1. **Full Action Scene Graph (FASG):** nodes correspond to the main/auxiliary action verb, objects, and their attributes, while edges encode semantic relationships between them.
2. **Gaze-Pruned Action Scene Graph (GPASG):** a compact subgraph containing only the interacted object and the gazed-at object.

Our pruning strategy can be viewed as improving global communication in the graph by shortening the longest path between key entities.

## Installation

Create environment
```bash
conda create -n ego_graphs python=3.10
conda activate ego_graphs
```

Install main requirements
```bash
pip install uv
cd /home/s3758869/egocentric_video_graph_framework_ar
uv pip install -r requirements.txt
```

Download spaCy language model
```bash
python -m spacy download en_core_web_sm
```

Install Open-GroundingDino requirements
```bash
uv pip install -r graph_construction/object_detection_pipeline/Open-GroundingDino/requirements.txt
```

```bash
cd graph_construction/object_detection_pipeline/Open-GroundingDino/models/GroundingDINO/ops
python setup.py build install
python test.py
```

**Note:** If `flash-attn` installation fails during step 3, install it separately after PyTorch (assuming you have a GPU and CUDA platform)
```bash
uv pip install torch torchvision torchaudio
uv pip install flash-attn --no-build-isolation
```

## Models

For GroundingDINO, we can use **GroundingDINO-T** finetuned on COCO (57.3 mAP) or the official pretrained checkpoints.

| Name | Pretrain Data | Task | mAP (COCO) | Checkpoint | Misc |
|-----|-----|-----|-----|-----|-----|
| GroundingDINO-T (official) | O365, GoldG, Cap4M | zero-shot | 48.4 | [GitHub](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) \| [HF](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth) | - |
| GroundingDINO-T (fine-tune) | O365, GoldG, Cap4M | finetune w/ COCO | 57.3 | [model](https://github.com/longzw1997/Open-GroundingDino/releases/download/v0.1.0/gdinot-coco-ft.pth) | [cfg](https://drive.google.com/file/d/1TJRAiBbVwj3AfxvQAoi1tmuRfXH1hLie/view?usp=drive_link) \| log |
| GroundingDINO-T (pretrain) | COCO, O365, LVIS, V3Det, GRIT-200K, Flickr30k (1.8M) | zero-shot | 55.1 | [model](https://github.com/longzw1997/Open-GroundingDino/releases/download/v0.1.0/gdinot-1.8m-odvg.pth) | [cfg](https://drive.google.com/file/d/1LwtkvBHkP1OkErKBsVfwjcedVXkyocA5/view?usp=drive_link) \| log |
| GroundingDINO-B (official) | COCO, O365, GoldG, Cap4M, OpenImage, ODinW-35, RefCOCO | zero-shot | 56.7 | [GitHub](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth) \| [HF](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth) | [cfg](graph_construction/object_detection_pipeline/Open-GroundingDino/tools/GroundingDINO_SwinB_cfg.py) |

Training configs and logs available here:  
https://github.com/longzw1997/Open-GroundingDino/tree/main?tab=readme-ov-file#training

Original GroundingDINO checkpoints:  
https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#luggage-checkpoints
