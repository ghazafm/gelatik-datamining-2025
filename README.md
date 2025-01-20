# Wildfire Detection with FLAME 3 Dataset

## Overview

This project utilizes the **FLAME 3** dataset to train a model for predicting wildfire presence and generating bounding boxes around areas of interest based on UAV-collected multi-spectral imagery. The dataset contains both thermal and visual data from prescribed burns, with images labeled for "Fire" and "No Fire" classes. The goal is to use this data for fire detection, segmentation, and to produce bounding boxes around detected fire regions.


## Objective
The primary objective is to develop a deep learning model capable of:

- **Detecting wildfire areas** in the provided imagery.
- **Predicting bounding boxes** around fire regions to localize the fire.


These bounding boxes can be used to assist in wildfire monitoring, providing valuable insights to response teams by pinpointing areas affected by fire.

## Dataset Description

### FLAME 3 Dataset

The **FLAME 3** dataset is focused on wildfire monitoring and is used to build Computer Vision (CV) models. The data includes:

- **Fire** and **No Fire** image pairs, consisting of:
  1. Raw RGB Image (Resolution: 4000x3000)
  2. Raw Thermal Image (Resolution: 640x512)
  3. Corrected FOV RGB Image (Resolution: 640x512)
  4. Thermal TIFF (Resolution: 640x512, Max Temperature ~500°C)

This dataset is collected from a prescribed fire at Sycan Marsh, Oregon, between 10/25/23 and 10/27/23.

### Dataset Folder Structure

- **CV Subset**: Contains fire and no-fire labeled image quartets.
- **NADIR Thermal Plot Set**: Thermal imagery captured from three plots (preburn, duringburn, postburn).

### Image File Types

- `.JPG` (RGB, Thermal, Corrected FOV)
- `.TIFF` (Thermal Temperature Raster)
- `.IRG` (Raw IR data)
- `.txt` (Ground Control Points)
- `.csv`, `.kml`, `.shp`, `.dbf` (Metadata)

### Tools and References

- Data processing tools can be found at the [FLAME Data Pipeline GitHub](https://github.com/BryceHopkins14/Flame-Data-Pipeline).
- [FLAME 3 Dataset Paper](https://arxiv.org/abs/2412.02831)

## Project Setup

### Prerequisites

- Python 3.x
- PyTorch
- OpenCV
- Matplotlib
- NumPy
- torchvision

### Installation

To install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Download

Download the **FLAME 3** dataset from [IEEE DataPort](https://ieeedataport.org).

### File Structure

```plaintext
.
├── .git/
├── data/
│   └── flame/
│       └── annotations.csv
│       └── images/
├── src/
│   └── notebook/
│       └── dataset/
│           └── flame.py
├── temp.py
└── README.md
```

## Usage

### Example Notebook: `example.ipynb`

In the example notebook, we demonstrate loading the **FLAME 3** dataset, applying transformations, and creating a DataLoader to handle the image batches. Below is a snippet for loading and visualizing an image batch:

```python
from dataset.flame import Flame
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np

# Define transformations
compose = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset
dataset = Flame(download=True, transform=compose)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Test DataLoader
for images, bboxes in loader:
    print("Image batch shape:", images.shape)
    print("Bounding box batch shape:", bboxes.shape)
    break
```

### Visualize the Image

To display an image and its bounding box:

```python
import matplotlib.pyplot as plt
import numpy as np

# Fetch a sample image
image, bbox = next(iter(loader))
image = image[0].permute(1, 2, 0).numpy()
image = np.clip(image, 0, 1)

# Display the image
plt.imshow(image)
plt.title(f"Bounding Box: {bbox[0]}")
plt.show()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use the FLAME 3 dataset, please cite the following paper:

```
Hopkins, B., ONeill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim I, Seielstad C, Afghah, F. (2024). FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management. arXiv preprint arXiv:2412.02831.
```

## Acknowledgments

- **NASA**, **NSF**, **Salt River Project**, and **DoD SERDP** funded this research.

## Contributors

We would like to thank the following contributors for their valuable work:

- **Fauzan Ghaza Madani** - Data Scientist
- **Muhammad Yasin Hakim** - Data Scientist
- **Fadhillah Hilmi** - Data Scientist
- **Other contributors** - See the [CONTRIBUTORS.md](CONTRIBUTORS.md) file for a full list.