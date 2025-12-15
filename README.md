# Synthetic Data + OCR

Simple data synthetic data generation and OCR training using CRNN based on https://github.com/SoyVitouPro/basicOCRonBacii and https://github.com/SoyVitouPro/gendatakh4ocr

## Preparation

- Create folders: `fonts/` (TTF/OTF files), `source_data/` (UTF-8 text files, one sample per line)
- Install deps (Python 3.9+): torch, pillow, numpy, tqdm, python-Levenshtein, tensorboard

## Generate synthetic data

```bash
	python generate_data.py
```

or

```bash
	python generate_data_batch.py
```

Outputs: images in output/images/, JSON per image in output/annotations/, consolidated metadata.json and labels.txt in output/.

## Training

- Config is in [train.py](train.py) `config` dict. Key fields: `labels_file`, `images_dir`, `output_dir`, `model_type` (`small`|`standard`), `img_height`, `hidden_size`, `batch_size`, `epochs`.
- Run:
  ```bash
  python train.py
  ```
- Creates `runs/.../checkpoints/` (latest, best, every 10 epochs) and `vocab.json`. TensorBoard logs in `runs/.../logs`.

## Inference

- Predict one image or a folder of images with a trained checkpoint.
  ```bash
  python inference.py --checkpoint runs/CRNN/checkpoints/checkpoint_best.pth \
  										--vocab runs/CRNN/vocab.json \
  										--image path/to/image.png
  # or
  python inference.py --checkpoint ... --vocab ... --images_dir path/to/dir --output preds.txt
  ```
