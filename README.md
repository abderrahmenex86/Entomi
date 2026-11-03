# Pesti

An insect and pest classifier trained on IP102 with MobileNetV3-Large. It
produces the weights that [FloraLens](https://github.com/abderrahmenex86/FloraLens)
runs on the phone.

## What it does

- Trains a 102-class classifier on IP102 with a MobileNetV3-Large backbone and a
  replaced classification head.
- Skips images that fail to decode, so one corrupt file in the dataset does not
  stop a run.
- Uses cross-entropy with 0.1 label smoothing to keep confidence calibrated.
- Schedules the learning rate with cosine annealing over AdamW.
- Reports top-1, top-5, and macro F1 on the validation split, and writes them to
  `training_history.json`.
- Exports the best checkpoint to `pest_model.pth`, plus a quantised ONNX model
  through `quantize.py` for the mobile app.

## Run it

```bash
git clone https://github.com/abderrahmeneex86/pesti
cd pesti
pip install -r requirements.txt
```

Download IP102 and lay it out as:

```text
ip102_v1.1/
├── images/
├── train.txt
├── val.txt
├── test.txt
└── classes.txt
```

Then:

```bash
python train.py        # trains, writes training_history.json and pest_model.pth
python inference.py    # runs a checkpoint over a folder of images
python quantize.py     # converts to ONNX for on-device use
```

A CUDA GPU is strongly recommended for training; inference runs on CPU.

## Results

No benchmark table is published. `training_history.json` records per-epoch
top-1, top-5, and macro F1, so the numbers can be read from a run, but the
repository does not carry a reference run to compare against.

## Stack

Python 3.10+ · PyTorch · TorchVision · TorchMetrics · pandas · Pillow · tqdm

## Related

- [FloraLens](https://github.com/abderrahmenex86/FloraLens) runs this model on
  Android
- [Flora](https://github.com/abderrahmenex86/flora) for plant classification
- [Segmenti](https://github.com/abderrahmenex86/segmenti) for disease segmentation
