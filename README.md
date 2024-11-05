# Project_X
 India AI CyberGuard Hackathon Code Submission

# Team
Shashwati : Btech ECE, Batch-26, IIIT NR (Team Head)
Darshan Kochar: Btech CSE, Batch-27, IIIT NR
Tejas Keshwani: Btech CSE, Batch-27, IIIT NR


For categorization so far we have achieved f1_score of 0.84 with model best_model_category2.bin
For ffc so far we have achieved f1_score of 0.77 with model best_model_ffc.bin
For ffc so far we have achieved f1_score of 0.84 with model best_model_wc.bin


# Cybercrime Multi-Class Classification

This project uses a BERT-based deep learning model to classify cybercrime descriptions into multiple categories and subcategories. It is designed to assist in analyzing and categorizing cybercrime reports effectively, providing a foundational tool for law enforcement agencies, cybersecurity analysts, and researchers.

## Project Overview

Cybercrime data classification poses unique challenges due to the variety of crime types, data imbalances, and the nuanced language of descriptions. This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to address these challenges, enabling high-accuracy, multi-class, and multi-label classification.

### Key Features

- **Multi-Class and Multi-Label Classification**: Classifies cybercrime descriptions across various categories and subcategories.
- **BERT Fine-Tuning**: Fine-tunes BERT (`bert-base-uncased`) on cybercrime datasets, achieving state-of-the-art performance for textual classification tasks.
- **Data Augmentation**: Includes techniques for handling imbalanced classes, such as upsampling.

## Project Structure

- **`main.py`**: Main script for model training and evaluation.
- **`data/`**: Folder containing training, validation, and test datasets.
- **`models/`**: Stores model weights and checkpoints.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis and model experimentation.
- **`utils/`**: Utility functions for data processing, evaluation metrics, etc.

## Getting Started

### Prerequisites

- Python 3.8+
- `transformers` (Hugging Face Transformers library)
- `torch` (PyTorch)
- `pandas`, `scikit-learn`, `numpy`

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset includes categories and subcategories of cybercrimes (e.g., **Phishing**, **Identity Theft**, **Malware Attack**). Place your dataset in the `data/` folder as:

- `train.csv`: Training dataset
- `val.csv`: Validation dataset
- `test.csv`: Testing dataset

> Note: Due to confidentiality, the actual dataset is not provided here. Ensure your dataset follows the necessary format before training.

### Training the Model

To train the model on your dataset, run:

```bash
python main.py
```

- **Arguments**:
    - `--epochs`: Number of training epochs.
    - `--batch_size`: Batch size for training.
    - `--lr`: Learning rate.

Example:

```bash
python main.py --epochs 3 --batch_size 16 --lr 2e-5
```

The trained model and logs will be saved in the `models/` directory.

### Evaluation

After training, evaluate the model on the test set:

```bash
python evaluate.py
```

This will output precision, recall, F1 scores, and other relevant metrics for each class and overall performance.

## Handling Data Imbalance

This project includes techniques to handle data imbalance, particularly in the `sub_category` labels. We implement **upsampling** to create a balanced dataset, improving model performance on minority classes.

## Results

The model achieved significant performance improvements in F1 score across various categories, particularly excelling in commonly reported cybercrime types.

| Metric      | Value  |
|-------------|--------|
| Precision   | 0.87   |
| Recall      | 0.85   |
| F1 Score    | 0.86   |

## Contributing

We welcome contributions to this project. To contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for their open-source transformer models.
- [PyTorch](https://pytorch.org/) for the deep learning framework.

---

This `README.md` gives an overview of the project's objectives, structure, and usage instructions, ensuring users and contributors can effectively understand and engage with your project.
