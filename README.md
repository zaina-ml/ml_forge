# ML Forge

A visual PyTorch pipeline editor. Build, train and run image classification models without writing code.
![ML Forge screenshot](assets/screenshot.png)
---

## What it does

- **Visual pipeline** - drag nodes onto a canvas, connect them with wires, and ML Forge generates and runs the training code for you
- **Three-tab workflow** - Data Prep → Model → Training
- **Live training** - watch loss curves update in real time, save checkpoints, run inference on your trained model
- **Export** - generate a clean, standalone `train.py` script you can run anywhere

---

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- torchvision

```
pip install torch torchvision dearpygui
```

GPU training is automatic if CUDA is available. CPU and Apple MPS are also supported.

---

## Quick start

```
git clone https://github.com/yourname/ml-forge
python main.py
```

Or use a template to skip setup:

**File → Templates → MNIST Classifier**

Then press **RUN** in the toolbar.

---

## Building your first model

### 1. Data Prep tab

- Add a **Dataset** node (MNIST, CIFAR10, CIFAR100, FashionMNIST, or ImageFolder)
- Chain **transforms**: ToTensor is required, add Normalize for best results
- End with a **DataLoader (train)** node
- For proper validation, add a second chain (same dataset with `train=False`) ending with **DataLoader (val)**

### 2. Model tab

- Start with an **Input** node — shape is auto-filled from your dataset
- Add layers: Linear, Conv2D, ReLU, BatchNorm2D, Flatten, Dropout, etc.
- End with an **Output** node — num classes is auto-filled from your dataset
- Connect nodes by dragging from an output pin to an input pin
- `in_features` and `in_channels` auto-fill when you connect layers
- After a Flatten node, the next Linear's `in_features` is calculated automatically

### 3. Training tab

Add these four nodes from the palette and wire them up:

```
DataLoaderBlock.images  →  ModelBlock.images
ModelBlock.predictions  →  Loss.pred
DataLoaderBlock.labels  →  Loss.target
Loss.loss               →  Optimizer.params
```

Configure epochs, device, checkpointing and early stopping in the right panel, then press **RUN**.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Del` | Delete selected nodes |
| `Ctrl+S` | Save project |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| Middle-drag | Pan the canvas |

---

## Supported datasets

| Dataset | Classes | Input shape |
|---------|---------|-------------|
| MNIST | 10 | 1 × 28 × 28 |
| FashionMNIST | 10 | 1 × 28 × 28 |
| CIFAR-10 | 10 | 3 × 32 × 32 |
| CIFAR-100 | 100 | 3 × 32 × 32 |
| ImageFolder | custom | 3 × 224 × 224 |

---

## Inference

After training, open **Run → Inference**, browse to your checkpoint (`.pth`), and click **Run Inference** to sample from the test set and see top-k predictions.

---

## Metrics

Open **Run → Metrics** (or click the **METRICS** button) to see a summary of your training run: final loss, best validation accuracy, fit diagnosis, and loss/accuracy curves.

---

## Saving and loading

Projects are saved as `.mlf` files (JSON). Use **File → Save / Save As** or `Ctrl+S`.

---

## Exporting code

**File → Export → Python → PyTorch** generates a standalone `train.py` that reproduces your pipeline. No ML Forge required to run it.

---

## Known limitations

- Sequential models only - skip connections and branching architectures are not supported in v1.0
- No learning rate schedulers in v1.0
- Tab sizing cosmetic bug - first tab (data) appears far larger than other tabs

---

## License

APACHE
