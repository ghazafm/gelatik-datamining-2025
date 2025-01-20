# Wildfire Detection with FLAME 3 Dataset

## **Table of Contents**

1. [Overview](#overview)
2. [Objective](#objective)
3. [Dataset Description](#dataset-description)
   - [FLAME 3 Dataset](#flame-3-dataset)
   - [Dataset Folder Structure](#dataset-folder-structure)
   - [Image File Types](#image-file-types)
4. [Project Setup](#project-setup)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Dataset Download](#dataset-download)
   - [File Structure](#file-structure)
5. [Framework Overview](#framework-overview)
   - [Folder Structure](#folder-structure)
   - [Key Components of the Framework](#key-components-of-the-framework)
   - [Example Workflow](#example-workflow)
6. [Usage](#usage)
   - [Example Notebook: `example.ipynb`](#example-notebook-exampleipynb)
   - [Visualize the Image](#visualize-the-image)
7. [License](#license)
8. [Citation](#citation)
9. [Acknowledgments](#acknowledgments)
10. [Contributors](#contributors)

## Overview

This project utilizes the **FLAME 3** dataset to train a model for predicting wildfire presence and generating bounding boxes around areas of interest based on UAV-collected multi-spectral imagery. The dataset contains both thermal and visual data from prescribed burns, with images labeled for "Fire" and "No Fire" classes. The goal is to use this data for fire detection, segmentation, and to produce bounding boxes around detected fire regions.

---

## Objective

The primary objective is to develop a deep learning model capable of:

- **Detecting wildfire areas** in the provided imagery.
- **Predicting bounding boxes** around fire regions to localize the fire.

These bounding boxes can be used to assist in wildfire monitoring, providing valuable insights to response teams by pinpointing areas affected by fire.

---

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

---

## Project Setup

### Prerequisites

- Python 3.x
- PyTorch
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

---

## Commit Message Guidelines

This project follows the **Conventional Commit** message format. The structure of the commit messages helps maintain a consistent and clear history. Here is the general structure for commit messages:

### Format
```
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

### Types of Commits
- **feat**: A new feature for the user.
- **fix**: A bug fix for the user.
- **docs**: Documentation changes.
- **style**: Changes that do not affect the meaning of the code (e.g., formatting).
- **refactor**: Refactoring code without changing functionality.
- **perf**: A performance improvement.
- **test**: Adding or modifying tests.
- **chore**: Other changes that don't modify the functionality (e.g., updates to build scripts, dependency management).
- **ci**: Continuous Integration changes.
- **build**: Changes that affect the build system or external dependencies.
- **revert**: Reverts a previous commit.

### Examples
- **feat(dataset): add support for new dataset class**
- **fix(training): correct learning rate for better convergence**
- **docs(readme): update project setup section**

### Subject
The subject should have the following structure:
```
<type>(<optional scope>): <subject-description>
```
For example, when adding a new endpoint to an API for creating coupons:
```
feat(api): Add support to create coupons
```

- The subject should be in lowercase.
- It should be concise, with a maximum of 100 characters.
- Do not end the subject with a period.


### Body
- The body provides additional context to explain why the change was made.
- It should be wrapped at 72 characters for readability.

### Footer
- The footer is optional and used for referencing issues or breaking changes.

---

## Framework Overview

This project is organized into a modular framework built to handle the **FLAME 3** dataset for wildfire detection. The folder structure is designed to separate concerns and allow for efficient management of data, scripts, and model training components.

### Folder Structure

- **`data/`**: Contains the raw data for both the **FLAME 3** and **ALBI** datasets. This includes images and annotations for training and evaluation.

  - **`flame/`**: Contains the FLAME 3 dataset, including image files (both RGB and Thermal) and annotations in CSV format.
  - **`albi/`**: Contains the ALBI dataset with similar structure as `flame/`.

- **`src/`**: Source code for preprocessing, model training, and evaluation.

  - **`notebook/`**: Jupyter notebooks for experimentation, data visualization, and model testing. Notebooks like `annotate.ipynb`, `ingesting.ipynb`, and `example.ipynb` help with dataset preparation and initial model development.
    - **`dataset/`**: Contains scripts for dataset loading and handling, such as `flame.py` and `albi.py`. These files define custom datasets that load the images, annotations, and apply transformations.
    - **`helper/`**: Contains utility scripts like `gdrive.py` for Google Drive access and `image_links.csv` for storing image metadata.

- **`temp/`**: Temporary scripts, such as `albi_temp.py`, for quick experimentation or testing specific components without modifying the core files.

- **`.git/`**: Git-related files and directories that manage version control for the project.

- **`README.md`**: The main documentation for the project.

### Key Components of the Framework

1. **Data Handling**:

   - Custom datasets like `Flame` and `Albi` are defined in the `dataset/` folder to load images, preprocess them, and pair them with corresponding annotations.
   - Transformations are applied to prepare data for training, such as resizing, normalization, and tensor conversion.

2. **Model Training and Evaluation**:

   - The main model architecture is developed using **PyTorch** and is designed to detect wildfire areas and predict bounding boxes.
   - Scripts for training the model, managing hyperparameters, and evaluating the performance are organized in a modular fashion to facilitate easy experimentation.

3. **Temporary Experiments**:
   - The `temp/` directory is used for testing scripts that may not be part of the final solution but are used for rapid prototyping or debugging.
4. **Model Inference**:
   - Once the model is trained, it can be used to predict the presence of wildfires and generate bounding boxes around detected areas. These predictions are visualized and evaluated to ensure the model's performance.

### Example Workflow

- Use the **`example.ipynb`** do your experiment
- When you call the dataset class(Flame) with class in **`dataset/`** scripts., your dataset will be stored in **`data/`**
- Train a model with **PyTorch** using custom datasets.
- Evaluate the model's performance using standard metrics, including the accuracy of bounding boxes and intersection-over-union (IoU).
- Save the model checkpoints for later use and inference.

This framework is designed to allow flexibility and scalability, enabling easy modifications to the dataset, model, and training pipeline.

---

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

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use the FLAME 3 dataset, please cite the following paper:

```
Hopkins, B., ONeill, L., Marinaccio, M., Rowell, E., Parsons, R., Flanary, S., Nazim I, Seielstad C, Afghah, F. (2024). FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management. arXiv preprint arXiv:2412.02831.
```

---

## Acknowledgments

- **NASA**, **NSF**, **Salt River Project**, and **DoD SERDP** funded this research.

---

## Contributors

We would like to thank the following contributors for their valuable work:

- **Fauzan Ghaza Madani** - Data Scientist
- **Muhammad Yasin Hakim** - Data Scientist
- **Fadhillah Hilmi** - Data Scientist
- **Other contributors** - See the [CONTRIBUTORS.md](CONTRIBUTORS.md) file for a full list.
