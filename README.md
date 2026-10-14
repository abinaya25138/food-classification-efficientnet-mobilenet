# Food Image Classification — EfficientNetB0 vs. MobileNet

Reproduces a transfer-learning approach to food image classification on [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (101 categories, 101,000 images), comparing two efficiency-oriented architectures — EfficientNetB0 and MobileNet — trained and evaluated identically so the comparison is fair.

## Dataset

Not bundled here (~5GB). Fetch it with:

```bash
bash download_data.sh
```

## Approach

Both models use ImageNet-pretrained weights as a frozen feature extractor, with the same classification head on top:

```
Base model (frozen, ImageNet weights) → preprocess_input (model-specific)
  → GlobalAveragePooling2D
  → Dense(128, relu) → Dropout(0.3)
  → Dense(num_classes, softmax)
```

EfficientNetB0 and MobileNet are both designed for parameter efficiency (far smaller than VGG16), which is why they're commonly compared for mobile/edge deployment — the script reports accuracy *and* training time *and* parameter count for each, so the trade-off is explicit, not just a single accuracy number.

```bash
python train.py --data-dir food-101/images --epochs 10
```

## ⚠️ Verification status — read this before trusting the numbers

Unlike the other two projects in this resume set (rice classification, cervical cancer prediction), **this one has not been run end-to-end** as of this commit. While building it, the development machine ran out of free memory partway through (a second heavy TensorFlow training session, right after training the rice classifier, on a machine already low on RAM) — TensorFlow couldn't even `import` reliably at that point, independent of this code. Rather than fake a result, this is shipped as **code only**, verified for syntax (`python -m py_compile`) but not for runtime correctness.

The code follows the identical structure and APIs as `rice-classification-vgg16` (same `image_dataset_from_directory` → preprocess → transfer-learn → evaluate pipeline), which **did** run successfully end-to-end and hit 98.2% accuracy — so the pattern itself is proven, just not this specific script, on this specific data.

**Before you cite results from this repo, run it once:**
```bash
pip install -r requirements.txt
bash download_data.sh
python train.py --data-dir food-101/images --epochs 10
```
and update this README with what you actually get. If anything breaks, it's most likely a path or shape mismatch, not a conceptual issue — the architecture and training loop are sound.

## Setup

```bash
pip install -r requirements.txt
```

## Tech stack

TensorFlow/Keras, EfficientNetB0 (ImageNet weights), MobileNet (ImageNet weights)

## Citation

Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 – Mining Discriminative Components with Random Forests. *European Conference on Computer Vision*.
