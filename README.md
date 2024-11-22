# India AI CyberGuard Hackathon Code Submission

## Team Members

| Name                | Program & Batch                  | Role                  | Contact                   |
|---------------------|----------------------------------|-----------------------|---------------------------|
| **Shashwati**       | B.Tech ECE, Batch-26, IIIT NR   | Team Head             | email@example.com         |
| **Darshan Kochar**  | B.Tech CSE, Batch-27, IIIT NR   | Developer & Researcher| darshankochar022@gmail.com|
| **Tejas Keshwani**  | B.Tech CSE, Batch-27, IIIT NR   | Developer & Analyst   | email@example.com         |

---

## Cybercrime Multi-Class Classification

This project presents a deep learning-based approach to classify cybercrime descriptions into multiple categories and subcategories, providing an efficient tool for law enforcement and cybersecurity analysts.

### **Highlights**
- **Multi-Class Classification**: Efficient categorization of cybercrime reports into primary categories and subcategories.
- **Streamlit App Integration**: A user-friendly interface for easy interaction and prediction.
- **BERT-Based Fine-Tuning**: Employs `bert-base-uncased` for classification tasks.
- **Addressing Imbalanced Data**: Implements upsampling to improve performance on minority classes.

---

## Project Overview

Cybercrime datasets often exhibit class imbalances, challenging language use, and unique categorizations. This project leverages **BERT (Bidirectional Encoder Representations from Transformers)** to build robust models for multi-class and multi-label classification.

---

## Hosted Models

The fine-tuned models are hosted on Hugging Face for public access:

[Darshan Kochar's Hugging Face Models](https://huggingface.co/Darshankochar022)

### **Model List**

| Model Name                         | Task                                           | Hugging Face Link                                 |
|------------------------------------|------------------------------------------------|--------------------------------------------------|
| **Category Classifier**            | Predict primary cybercrime category           | [Category Classifier](https://huggingface.co/Category-Model) |
| **Financial Fraud Classifier**     | Specialized in financial fraud classification | [Financial Fraud Classifier](https://huggingface.co/Financial-Model) |
| **Women and Child Classifier**     | Crimes affecting women and children           | [Women and Child Classifier](https://huggingface.co/WC-Model) |
| **Other Cyber Crimes Classifier**  | Handles all other crime categories            | [Other Cyber Crime Classifier](https://huggingface.co/OCC-Model) |

---

## Project Structure

- **`main.py`**: Script for training and evaluating models.
- **`data/`**: Contains `train.csv`, `val.csv`, and `test.csv`.
- **`models/`**: Stores model checkpoints and weights.
- **`notebooks/`**: For exploratory data analysis and experiments.
- **`utils/`**: Contains utility functions for preprocessing and evaluation.

---

## Getting Started

### **Prerequisites**

- **Python 3.8+** (Anaconda recommended)
- Libraries: `transformers`, `torch`, `pandas`, `scikit-learn`, `numpy`

### **Setup Instructions**

#### Step 1: Virtual Environment Setup
```bash
conda create -n cyberguard python=3.8
conda activate cyberguard


# India AI CyberGuard Hackathon Code Submission

## Team
- **Shashwati**: B.Tech ECE, Batch-26, IIIT NR (Team Head)
- **Darshan Kochar**: B.Tech CSE, Batch-27, IIIT NR
- **Tejas Keshwani**: B.Tech CSE, Batch-27, IIIT NR

## Cybercrime Multi-Class Classification

This project uses a BERT-based deep learning model to classify cybercrime descriptions into multiple categories and subcategories. It assists in analyzing and categorizing cybercrime reports effectively, providing a tool for law enforcement agencies, cybersecurity analysts, and researchers.

### Project Overview

Cybercrime data classification poses unique challenges due to the variety of crime types, data imbalances, and nuanced language. This project uses **BERT (Bidirectional Encoder Representations from Transformers)** to enable high-accuracy, multi-class, and multi-label classification.

### Key Features

- **Multi-Class and Multi-Label Classification**: Classifies cybercrime descriptions across various categories and subcategories.
- **BERT Fine-Tuning**: Fine-tunes BERT (`bert-base-uncased`) on cybercrime datasets.
- **Data Augmentation**: Includes techniques for handling imbalanced classes, such as upsampling.

## Hosted Models

The models trained for this project are hosted on Hugging Face and can be accessed via the following link:

[Darshan Kochar's Hugging Face Models](https://huggingface.co/Darshankochar022)

### Available Models

- **Category Prediction Model** (`best_model_category2.bin`): Predicts the primary category of cybercrime.
- **Financial Fraud Crime Model** (`best_model_ffc.bin`): Focused on classifying financial fraud-related cybercrime.
- **Women and Child Related Crime Model** (`best_model_wc.bin`): Specializes in categorizing cybercrime affecting women and children.
- **Other Cyber Crime Model** (`best_model_occ.bin`): Covers additional cybercrime categories.

## Project Structure

- **`main.py`**: Main script for model training and evaluation.
- **`data/`**: Folder containing training, validation, and test datasets.
- **`models/`**: Stores model weights and checkpoints.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis and model experimentation.
- **`utils/`**: Utility functions for data processing, evaluation metrics, etc.

## Getting Started

### Prerequisites

- **Python 3.8+** (Anaconda recommended)
- `transformers` (Hugging Face Transformers library)
- `torch` (PyTorch)
- `pandas`, `scikit-learn`, `numpy`

#### Step 1: Create a Virtual Environment
It’s recommended to create a virtual environment to manage dependencies. To do this with Anaconda:

```bash
conda create -n cyberguard python=3.8
conda activate cyberguard
```

#### Step 2: Install Dependencies
After activating the environment, install the required packages:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset includes categories and subcategories of cybercrimes (e.g., **Phishing**, **Identity Theft**, **Malware Attack**). Place your dataset in the `data/` folder as:

- `train.csv`: Training dataset
- `val.csv`: Validation dataset
- `test.csv`: Testing dataset

> **Note**: Due to confidentiality, the actual dataset is not provided here. Ensure your dataset follows the necessary format before training.

### Running the Project

#### To Train the Model

Run the following command to start model training:

```bash
EDA.ipynb
```
```bash
Copy.ipynb
```
```bash
Mains.ipynb
```

- **Arguments**:
    - `--epochs`: Number of training epochs.
    - `--batch_size`: Batch size for training.
    - `--lr`: Learning rate.

Example:

```bash
python main.py --epochs 3 --batch_size 16 --lr 2e-5
```
### For Evaluators

#### Running the Streamlit Interface

To execute the code using a Streamlit interface, run:

```bash
streamlit run x.py
```

## Handling Data Imbalance

This project includes techniques to handle data imbalance, particularly in the `sub_category` labels. We implement **upsampling** to create a balanced dataset, improving model performance on minority classes.

## Model Performance

The table below summarizes the performance of different models:

| Model                                  | Precision | Recall | F1 Score | Accuracy |
|----------------------------------------|-----------|--------|----------|----------|
|  Category Classifier  | 0.9002    | 0.9014 | 0.9000   | 0.9017   |
|  Financial Fraud Classifier          | 0.7772      | 0.7809   | 0.7781     | 0.7732     |
|  Other Cyber Crime Classifier          | 0.6672      | 0.6889   | 0.6876    | 0.7264     |
|  Women/ Child Classifier       | 0.9704    | 0.9806 | 0.9804   | 0.9892   |

These models are evaluated based on precision, recall, F1 score, and accuracy for the task of classifying cybercrime descriptions.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for their open-source transformer models.
- [PyTorch](https://pytorch.org/) for the deep learning framework.

