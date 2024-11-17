import streamlit as st
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score  # Import necessary metrics

label_to_category = {
    0: "Financial Fraud Crime",
    1: "Other Cyber Crime",
    2: "Women & Child Related Crime"
}

category_model = BertForSequenceClassification.from_pretrained("Darshankochar022/cyberguard_BERT_Category")
category_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

financial_fraud_subcategories = {
    0: "Business Email Compromise /Email Takeover",
    1: "Debit/Credit Card Fraud/Sim Swap Fraud",
    2: "Dematdepository Fraud",
    3: "E Wallet Related Fraud",
    4: "Fraud Callvishing",
    5: "Internet Banking Related Fraud",
    6: "UPI Related Frauds"
}

women_child_related_subcategories = {
    0: "Child Pornography CP/Child Sexual Abuse Material CSAM",
    1: "Rape/ GangRape RGR Sexually Abusive Content",
    2: "Sexually Explicit Act",
    3: "Sexually Obscene Material"
}

other_cyber_crime_subcategories = {
    0: "Cheating by Impersonation",
    1: "Cryptocurrency Fraud",
    2: "Cyber Bullying / Stalking / Sexting",
    3: "Cyber Terrorism",
    4: "Damage to Computer Systems",
    5: "Data Breach/Theft",
    6: "DoS/DDoS Attacks",
    7: "Email Hacking",
    8: "Email Phishing",
    9: "Fake Impersonating Profile",
    10: "Hacking/Defacement",
    11: "Impersonating Email",
    12: "Intimidating Email",
    13: "Malware Attack",
    14: "Online Gambling / Betting",
    15: "Online Job Fraud",
    16: "Online Matrimonial Fraud",
    17: "Online Trafficking",
    18: "Other",
    19: "Profile Hacking/Identity Theft",
    20: "Provocative Speech for Unlawful Acts",
    21: "Ransomware",
    22: "Ransomware Attack",
    23: "SQL Injection",
    24: "Tampering with Computer Source Documents",
    25: "Unauthorized Access/Data Breach",
    26: "Website Defacement/Hacking"
}

financial_fraud_model = BertForSequenceClassification.from_pretrained("Darshankochar022/cyberguard_BERT_Financial_Fraud_Crime")
women_child_related_model = BertForSequenceClassification.from_pretrained("Darshankochar022/cyberguard_BERT_WomenChild_Related_Crimes")
other_cyber_crime_model = BertForSequenceClassification.from_pretrained("Darshankochar022/cyberguard_BERT_Other_Cyber_Crime")

subcategory_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Streamlit app title and input sections
st.title("CyberCrime Classifier")
st.subheader("Enter Text Directly")
user_text = st.text_area("Type your text here:")

st.subheader("Or Upload a CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Category classification function
def classify_category(text):
    inputs = category_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = category_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_to_category.get(prediction, "Unknown Category")

# Subcategory classification function
def classify_subcategory(text, category):
    if category == "Financial Fraud Crime":
        model = financial_fraud_model
        subcategory_map = financial_fraud_subcategories
    elif category == "Women & Child Related Crime":
        model = women_child_related_model
        subcategory_map = women_child_related_subcategories
    elif category == "Other Cyber Crime":
        model = other_cyber_crime_model
        subcategory_map = other_cyber_crime_subcategories
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
# Process uploaded CSV file
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Check for required columns in the uploaded CSV
    if "crimeaditionalinfo" not in df.columns or "category" not in df.columns or "sub_category" not in df.columns:
        st.error("CSV file must contain 'crimeaditionalinfo', 'category', and 'sub_category' columns.")
    else:
        st.write("CSV File Contents:")
        st.dataframe(df)

        st.markdown("### Processing CSV File...")

        # Classify each row's text and store results
        results = []
        for index, row in df.iterrows():
            text = row["crimeaditionalinfo"]
            category = classify_category(text)
            subcategory = classify_subcategory(text, category) if category != "Unknown Category" else "No Subcategory"
            results.append({"text": text, "category": category, "subcategory": subcategory})

        # Convert results to DataFrame and merge with actual labels
        results_df = pd.DataFrame(results)
        results_df["category"] = df["category"]
        results_df["sub_category"] = df["sub_category"]
        st.write("Final Output from CSV Processing:")
        st.dataframe(results_df)
        
        # Calculate and display metrics for categories
        cat_precision, cat_recall, cat_f1, _ = precision_recall_fscore_support(
            results_df["category"], results_df["category"], average="weighted"
        )
        cat_accuracy = accuracy_score(results_df["category"], results_df["category"])

        st.markdown("### Category Classification Metrics")
        st.write(f"Precision: {cat_precision:.2f}")
        st.write(f"Recall: {cat_recall:.2f}")
        st.write(f"F1 Score: {cat_f1:.2f}")
        st.write(f"Accuracy: {cat_accuracy:.2f}")

        # Calculate and display metrics for subcategories
        sub_precision, sub_recall, sub_f1, _ = precision_recall_fscore_support(
            results_df["sub_category"], results_df["subcategory"], average="weighted"
        )
        sub_accuracy = accuracy_score(results_df["sub_category"], results_df["subcategory"])

        st.markdown("### Subcategory Classification Metrics")
        st.write(f"Precision: {sub_precision:.2f}")
        st.write(f"Recall: {sub_recall:.2f}")
        st.write(f"F1 Score: {sub_f1:.2f}")
        st.write(f"Accuracy: {sub_accuracy:.2f}")

        # Optionally save results to a CSV file
        results_df.to_csv("output_with_subcategories.csv", index=False)
        st.success("Classification complete. Results saved to output_with_subcategories.csv.")

else:
    st.write("Please enter text or upload a CSV file to proceed.")
