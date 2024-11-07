import streamlit as st
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Define category and subcategory mappings
label_to_category = {
    0: "Financial Fraud Crime",
    1: "Other Cyber Crime",
    2: "Women & Child Related Crime"
}

# Load category classification model and tokenizer
category_model = BertForSequenceClassification.from_pretrained("Darshankochar022/cyberguard_BERT_Category")
category_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Subcategory mappings for Financial Fraud and Women & Child Related Crime
financial_fraud_subcategories = {
    0: "Business Email Compromise /Email Takeover",
    1: "Debit/Credit Card Fraud/Sim Swap Fraud",
    2: "Dematdepository Fraud",
    3: "E Wallet Related Fraud",
    4: "Fraud Callvishing",
    5: "Internet Banking Related Fraud",
    7: "UPI Related Frauds",
}

women_child_related_subcategories = {
    0: "Child Pornography CP/Child Sexual Abuse Material CSAM",
    1: "Rape/ GangRape RGR Sexually Abusive Content",
    2: "Sexually Explicit Act",
    3: "Sexually Obscene Material"
}

# Load subcategory models
financial_fraud_model = BertForSequenceClassification.from_pretrained("Darshankochar022/cyberguard_BERT_Financial_Fraud_Crime")
women_child_related_model = BertForSequenceClassification.from_pretrained("Darshankochar022/cyberguard_BERT_WomenChild_Related_Crimes")

subcategory_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Streamlit app title and input sections
st.title("CyberCrime Classifier")
st.subheader("Enter Text Directly")
user_text = st.text_area("Type your text here:")

st.subheader("Or Upload a CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Category classification function
def classify_category(text):
    if isinstance(text, str):
        inputs = category_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = category_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return label_to_category.get(prediction, "Unknown Category")
    else:
        return "Invalid input. Please provide a valid string."

# Subcategory classification function
def classify_subcategory(text, category):
    if category == "Financial Fraud Crime":
        model = financial_fraud_model
        subcategory_map = financial_fraud_subcategories
    elif category == "Women & Child Related Crime":
        model = women_child_related_model
        subcategory_map = women_child_related_subcategories
    else:
        return "No subcategory defined for this category."

    inputs = subcategory_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    subcategory_prediction = torch.argmax(outputs.logits, dim=1).item()
    return subcategory_map.get(subcategory_prediction, "Unknown Subcategory")

# Process user input text
if user_text:
    st.write("You entered the following text:")
    st.write(user_text)
    st.markdown("### Processing Text...")

    category = classify_category(user_text)
    st.write(f"Predicted Category: {category}")

    if category != "Unknown Category":
        subcategory = classify_subcategory(user_text, category)
        st.write(f"Subcategory: {subcategory}")

# Process uploaded CSV file
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Check if the expected column is present
    if "crimeaditionalinfo" not in df.columns:
        st.error("CSV file must contain a column named 'crimeaditionalinfo' with text data.")
    else:
        st.write("CSV File Contents:")
        st.dataframe(df)

        st.markdown("### Processing CSV File...")

        # Classify each row's text and store results
        results = []
        for index, row in df.iterrows():
            text = row["crimeaditionalinfo"]
            category = classify_category(text)
            subcategory = classify_subcategory(text, category) if category in ["Financial Fraud Crime", "Women & Child Related Crime"] else "No Subcategory"
            results.append({"text": text, "category": category, "subcategory": subcategory})

        # Display and save the results
        results_df = pd.DataFrame(results)
        st.write("Final Output from CSV Processing:")
        st.dataframe(results_df)
        
        # Optionally save results to a CSV file
        results_df.to_csv("output_with_subcategories.csv", index=False)
        st.success("Classification complete. Results saved to output_with_subcategories.csv.")

else:
    st.write("Please enter text or upload a CSV file to proceed.")
