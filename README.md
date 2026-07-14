# G3Ego: Gaze-Guided Graphs for Egocentric Action Understanding

G3Ego builds compact action scene graphs from egocentric video. It uses gaze as a structural cue: from sparsely sampled frames, the pipeline creates vision-language descriptions, grounds objects and hands, prunes the graph around the camera wearer's gaze, and temporally aggregates graph embeddings for action recognition and anticipation.

The main setup uses Gaze-Guided Graphs with 32 sampled frames per clip, RGB frame features, hand features, and object grounding features.

## Installation

Create the conda environment first.

```bash
conda create -n ego_graphs python=3.10 -y
conda activate ego_graphs
pip install uv
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

GroundingDINO has its own requirements and CUDA ops.

```bash
uv pip install -r graph_construction/object_detection_pipeline/Open-GroundingDino/requirements.txt
cd graph_construction/object_detection_pipeline/Open-GroundingDino/models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../../../../..
```

If `flash-attn` fails, install PyTorch first and then install it without build isolation.

```bash
uv pip install torch torchvision torchaudio
uv pip install flash-attn --no-build-isolation
```

## G3Ego Pipeline

Generate per-frame scene descriptions with Qwen3-VL. The EGTEA config samples up to 32 frames per clip and generates short action text with 160 new tokens.

```bash
python vlm_annotation/egtea_vlm_annotation_pipeline.py \
  --config vlm_annotation/qwen35vl/configs/qwen3_vl_32b_instruct_egtea.json
```

Each frame is annotated independently. The prompt constrains the output to one sentence that starts with `The camera wearer is`, uses an `-ing` verb, and focuses on the camera wearer's hands and the manipulated or attended object.

VLM inference is deterministic. Qwen3-VL-32B-Instruct is loaded in FP16, with no 4-bit or 8-bit quantization in the provided EGTEA config. Images are resized with a thumbnail limit of `448 x 448`, encoded through the Hugging Face `AutoProcessor` chat template, and decoded with greedy generation: `do_sample=False`, `max_new_tokens=160`, default `num_beams=1`, and special tokens skipped during decoding. Since sampling is disabled, temperature/top-p are not used.

Parse the descriptions into graph fields. This writes `parse_annotation.csv` files and the global vocabularies used by graph construction.

```bash
python graph_construction/vlm_annotation_parse/qwen35vl_egtea_parse.py \
  --input-root /path/to/vlm_ann_Qwen3-VL-32B-Instruct \
  --overwrite
```

Run frame-based vision encoder inference with DINOv3-L/16. These RGB descriptors are stored per clip as `frame_features_model_dinov3_vitl16.h5`.

```bash
python graph_construction/feature_extraction/dinov3_egtea.py \
  --videos-root /path/to/framewise_videos \
  --max-frames-per-clip 32 \
  --model-id facebook/dinov3-vitl16-pretrain-lvd1689m
```

Ground objects and hands with GroundingDINO-T. This attaches object features, the gazed-at object, and hand features when `--extract-hands` is enabled.

```bash
python graph_construction/object_detection_pipeline/Open-GroundingDino/extract_gdino_features_egtea.py \
  --config_file graph_construction/object_detection_pipeline/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py \
  --model_checkpoint_path /path/to/groundingdino_swint_ogc.pth \
  --annotation-root /path/to/vlm_ann_Qwen3-VL-32B-Instruct \
  --frames-root /path/to/framewise_videos \
  --extract-hands
```

Populate the parsed annotations with the grounded gaze target. This is the gaze-guided graph pruning signal used by G3Ego.

```bash
python graph_construction/object_detection_pipeline/get_gazed_at_objects_egtea.py \
  --input-root /path/to/vlm_ann_Qwen3-VL-32B-Instruct
```

Train G3Ego with temporal graph aggregation. The gaze-guided configs combine RGB, hand, and object features.

EGTEA Gaze+ action recognition is reported across the three official splits. The per-split configs use 32 graphs per clip.

```bash
python graph_training/train_graph_lstm_egtea.py \
  --config-path graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned/split1.json

python graph_training/train_graph_lstm_egtea.py \
  --config-path graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned/split2.json

python graph_training/train_graph_lstm_egtea.py \
  --config-path graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned/split3.json
```

MECCANO action recognition uses the gaze-pruned graph configuration.

```bash
python graph_training/train_graph_mlp_meccano.py \
  --config-path graph_training/configs/meccano/action_recognition/gpasg_run_config.json
```

MECCANO action anticipation uses the gaze-pruned next-action LSTM configuration.

```bash
python graph_training/train_graph_lstm_meccano.py \
  --config-path graph_training/configs/meccano/next_action_prediction/gpasg_next_action_lstm.json \
  --objective-metric f1
```

## Models Used

| Model | Link | Config |
| --- | --- | --- |
| Qwen3-VL-32B-Instruct | [Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | [`qwen3_vl_32b_instruct_egtea.json`](vlm_annotation/qwen35vl/configs/qwen3_vl_32b_instruct_egtea.json) |
| DINOv3-L/16 | [Hugging Face](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `--model-id facebook/dinov3-vitl16-pretrain-lvd1689m` |
| GroundingDINO-T | [GitHub](https://github.com/IDEA-Research/GroundingDINO) | [`GroundingDINO_SwinT_OGC.py`](graph_construction/object_detection_pipeline/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py) |
