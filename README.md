# India AI CyberGuard Hackathon Code Submission

## Team Members

| Name                | Program & Batch                  | Role                  | 
|---------------------|----------------------------------|-----------------------|
| **Shashwati**       | B.Tech ECE, Batch-26, IIIT NR   | Team Head             | 
| **Darshan Kochar**  | B.Tech CSE, Batch-27, IIIT NR   | Developer & Researcher| 
| **Tejas Keshwani**  | B.Tech CSE, Batch-27, IIIT NR   | Developer & Analyst   | 

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
| **Category Classifier**            | Predict primary cybercrime category           | [Category Classifier](https://huggingface.co/Darshankochar022)) |
| **Financial Fraud Classifier**     | Specialized in financial fraud classification | [Financial Fraud Classifier](https://huggingface.co/Darshankochar022)) |
| **Women and Child Classifier**     | Crimes affecting women and children           | [Women and Child Classifier](https://huggingface.co/Darshankochar022))  |
| **Other Cyber Crimes Classifier**  | Handles all other crime categories            | [Other Cyber Crime Classifier](https://huggingface.co/Darshankochar022))  |

---


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

```

#### Step 2: Install Dependencies
After activating the environment, install the required packages:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset includes categories and subcategories of cybercrimes (e.g., **Phishing**, **Identity Theft**, **Malware Attack**). Place your dataset in the official website and those csvs made for trainig are made after preprocessing but due to larger size can't be uploaded:

- `train.csv`: Training dataset
- 'financial.csv' : subset of original
- 'women_child.csv':subset of original
- 'other.csv':subset of original 
- `test.csv`: Testing dataset

> **Note**: Due to confidentiality, the actual dataset is not provided here. Ensure your dataset follows the necessary format before training.

### Running the Project

#### To Train the Model

Run the following command to start model training:

```bash
EDA.ipynb
```
```bash
Category.ipynb
ffc.ipynb
wcc.ipynb
occ.ipynb
```
```bash
streamlit run App.py
```

- **Arguments**:
    - `--epochs`: Number of training epochs.
    - `--batch_size`: Batch size for training.
    - `--lr`: Learning rate.

### For Evaluators

#### Running the Streamlit Interface

To execute the code using a Streamlit interface,clone the repo and  run:

```bash
pip install-r requirements.txt
streamlit run x.py
```

## Handling Data Imbalance

This project includes techniques to handle data imbalance, particularly in the `sub_category` labels. We implement **upsampling** to create a balanced dataset, improving model performance on minority classes.

## Model Performance

The table below summarizes the performance of different models:

| Model                                  | Precision | Recall | F1 Score | Accuracy |
|----------------------------------------|-----------|--------|----------|----------|
|  Category Classifier  | 0.9342    | 0.9337 | 0.9342   | 0.9337   |
|  Financial Fraud Classifier          | 0.9296     | 0.9283   | 0.9296     | 0.9280     |
|  Other Cyber Crime Classifier  + ('all-mp-net' from sentence_transformers)       | 0.8880      | 0.8850   | 0.8880    | 0.8851     |
|  Women/ Child Classifier       | 0.9704    | 0.9806 | 0.9804   | 0.9892   |

Although the model other cyber crime is not so efficient bt its effficiency has increased simultaneously after trying to run it with similarity search using chroma_db, we highly encourage everyone to try the user interface and test us


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for their open-source transformer models.
- [PyTorch](https://pytorch.org/) for the deep learning framework.

