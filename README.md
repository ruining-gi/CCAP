## Models (Baseline Implementations)

This repository currently provides **baseline segmentation model definitions only** (no training / testing / dataset scripts). The implementations are intended to be imported into your own pipeline for training and evaluation on CCAP (masks) + ARCADE (images).

### Available model files

- `models/unet.py` — U-Net baseline  
- `models/deeplabv3.py` — DeepLabV3 baseline  
- `models/enet.py` — ENet baseline  
- `models/gcn.py` — GCN-based segmentation baseline  
- `models/sk.py` — SK-based module/model variant (see code comments for details)

> Note: The exact class names may differ by file. Please open each file to confirm the exported class (e.g., `UNet`, `DeepLabV3`, etc.).

### How to use with CCAP + ARCADE

CCAP releases masks and split CSV mapping files (`train.csv`, `val.csv`, `test.csv`) that map each mask to the corresponding ARCADE image using two columns:

- `mask_filename` (path inside extracted CCAP directory)
- `source_image_relpath` (path inside ARCADE directory, e.g., `/train/506.png`)

**Recommended path joining:**

- CCAP mask path: `<CCAP_ROOT>/<mask_filename>`
- ARCADE image path: `<ARCADE_ROOT>/<source_image_relpath>`  
  If `source_image_relpath` starts with `/`, treat it as relative by removing the leading slash.

You can build a dataset loader in your own code by reading the CSV and returning `(image, mask)` pairs after preprocessing (resize/crop, normalize, etc.). Then import a model from `models/` and train with your preferred loss (e.g., BCE/Dice for binary masks).

### Minimal sanity check (import + forward pass)

If you want to quickly verify that the model code imports correctly and runs a forward pass, use the snippet below.  
First confirm the class name in each model file (e.g., `class UNet(...)`), then update the imports accordingly.

```bash
python - << 'PY'
import torch

# Update these imports/class names to match your code
from models.unet import UNet
# from models.deeplabv3 import DeepLabV3
# from models.enet import ENet
# from models.gcn import GCN
# from models.sk import SKNet

x = torch.randn(1, 3, 512, 512)  # dummy input
m = UNet()                       # change to another model if needed
m.eval()

with torch.no_grad():
    y = m(x)

if isinstance(y, dict):
    print("Output is a dict:", {k: tuple(v.shape) for k, v in y.items()})
else:
    print("Output shape:", tuple(y.shape))
PY
