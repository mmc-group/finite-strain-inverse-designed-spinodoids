﻿This repository provides the data and codes for the paper: **Experiment-informed finite-strain inverse design of spinodal metamaterials**, Extreme Mechanics Letters 74 (2025) 102274.

Link to paper: [https://doi.org/10.1016/j.eml.2024.102274](https://doi.org/10.1016/j.eml.2024.102274)

---

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Testing](#testing)
  - [Optimization](#optimization)
- [License](#license)

---

## Overview

The code includes functionality for:

1. **Geometry generation**: MATLAB scripts for creating periodic spinodoid structures and exporting them as `.stl` files for further analysis or fabrication.
2. **Model Training**: Training the Partial Input Convex Neural Network (PICNN) based framework to predict stress-strain behavior based on specified design parameters.
3. **Testing and plotting**: Tools to evaluate the trained model's performance by comparing predicted and ground-truth stress-strain curves, alongside visualization capabilities.
4. **Optimization**: Optimization framework to inverse design spinodoid structures for a desired finite-strain response.

---

## Dependencies

Ensure you have the following dependencies installed:

```bash
pip install torch numpy pandas scipy matplotlib scikit-learn
```

---

## Usage

#### Dataset

The dataset is structured as follows:

```
drivers/data/
│-- X/
│   ├── 000021_tes_X.csv
│   ├── 233456_tes_X.csv
│-- Y/
│   ├── 345067_tes_Y.csv
│-- Z/
│   └── 456069_tes_z.csv
```

With the following naming convention:
`[theta_1][theta_2][theta_3]_tes_[direction].csv`

Each CSV file has three columns:
1. **Strain (eps)** – First column
2. **Stress (sigma)** – Second column

#### Training
To train the model, run:

```bash
python main.py train [direction=[X|Y|Z]] [improve_checkpoint=True|False]
```

Example: Training the model in the Y direction for the first time without improving a checkpoint model:

```bash
python main.py train Y False
```

#### Testing
To test the model, use:

```bash
python main.py test [direction=[X|Y|Z]]
```

Example: Testing the model predictions in the Y direction:

```bash
python main.py test Y
```

#### Optimization
To perform optimization, run:

```bash
python main.py opt [direction=[X|Y|Z]]
```

Example: Constructing target curves from the dataset in X-direction by multiplying the stress-responses by `query_factor` and, subsequently, running an optimization loop:

```bash
python main.py opt X
```

## Authors:
This software is written by Prakash Thakolkaran and Siddhant Kumar. It is based on the work published in the [Extreme Mechanics Letter](https://www.sciencedirect.com/science/article/pii/S2352431624001548) by:

- Prakash Thakolkaran
- Michael Espinal
- Somayajulu Dhulipala
- Siddhant Kumar
- Carlos M. Portela

## License

This code is provided under the MIT License.

---
