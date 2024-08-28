
# Human Activity Recognition using CNN and Transformer

This repository contains a deep learning approach for Human Activity Recognition (HAR) using a combination of Convolutional Neural Networks (CNN) and Transformers.

## Overview

Human Activity Recognition (HAR) is a challenging task, where the goal is to classify different activities based on sensor data. This project implements a hybrid model leveraging CNNs for feature extraction and Transformers for sequence modeling.

## Features

- **CNN-Based Feature Extraction:** Efficiently captures spatial features from the sensor data.
- **Transformer-Based Sequence Modeling:** Utilizes self-attention mechanisms to model temporal dependencies in the activity sequences.
- **Modular Code Structure:** Easy to extend and adapt for various HAR datasets.

## Requirements

To run this project, ensure you have Python installed along with the required libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/Sk-22-2-11/HAR-CNN-Transformer.git
    ```

2. Navigate to the project directory:

    ```bash
    cd HAR-CNN-Transformer
    ```

3. Train the model using the dataset:

    ```bash
    python train.py --model [Your Model Path] --dataset [Your Dataset Path]
    ```

4. Evaluate the model:

    ```bash
    python val.py --model [Your Model Path] --dataset [Your Dataset Path]
    ```

## Dataset

The model is compatible with various HAR datasets, such as the [NTU-Fi HAR Dataset, UT HAR Dataset, and WIDAR Dataset](https://drive.google.com/drive/folders/13qxmFQ-h8ei2m7EbBQOCxCHZJ_cDJGFY?usp=sharing). Make sure to preprocess the dataset accordingly before feeding it into the model.

## Results

| Model      | Accuracy | Flops(M) |
|------------|----------|----------|
| CNN + Transformer | 80.36%   | 4.72   |

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Sk-22-2-11/HAR-CNN-Transformer/blob/main/LICENSE) file for details.
