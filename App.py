import streamlit as st
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

# Define category labels
label_to_category = {
    0: "Financial Fraud Crime",
    1: "Other Cyber Crime",
    2: "Women & Child Related Crime"
}

# Load category classifier
category_model = BertForSequenceClassification.from_pretrained("Darshankochar022/Category_Classifier")
category_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Financial fraud subcategories
financial_fraud_subcategories = {
    2: "Debit/Credit Card Fraud",
    7: "SIM Swap Fraud",
    6: "Internet Banking-Related Fraud",
    1: "Business Email Compromise/Email Takeover",
    4: "E-Wallet Related Frauds",
    5: "Fraud Call/Vishing",
    3: "Demat/Depository Fraud",
    8: "UPI-Related Frauds",
    0: "Aadhaar Enabled Payment System (AEPS) Fraud"
}

# Women & child-related subcategories
women_child_related_subcategories = {
    0: "Child Pornography CP/Child Sexual Abuse Material CSAM",
    1: "Rape/GangRape RGR Sexually Abusive Content",
    3: "Sale, Publishing, and Transmitting Obscene Material/Sexually Explicit Material"
}

# Other cyber crime subcategories
other_cyber_crime_subcategories = [
    "Email Phishing", "Cheating by Impersonation", "Fake/Impersonating Profile", "Profile Hacking/Identity Theft",
    "Provocative Speech of Unlawful Acts", "Impersonating Email", "Intimidating Email", "Online Matrimonial Fraud",
    "Cyber Bullying/Stalking/Sexting", "Email Hacking", "Damage to Computer Systems", "Tampering with Computer Source Documents",
    "Defacement/Hacking", "Unauthorized Access/Data Breach", "Online Cyber Trafficking", "Online Gambling/Betting Fraud",
    "Ransomware", "Cryptocurrency Crime", "Cyber Terrorism", "Any Other Cyber Crime",
    "Targeted scanning/probing of critical networks/systems", "Compromise of critical systems/information",
    "Unauthorized access to IT systems/data", "Defacement of websites or unauthorized changes, such as inserting malicious code or external links",
    "Malicious code attacks (e.g., virus, worm, Trojan, Bots, Spyware, Ransomware, Crypto miners)",
    "Attacks on servers (Database, Mail, DNS) and network devices (Routers)",
    "Identity theft, spoofing, and phishing attacks", "Denial of Service (DoS) and Distributed Denial of Service (DDoS) attacks",
    "Attacks on critical infrastructure, SCADA, operational technology systems, and wireless networks",
    "Attacks on applications (e.g., E-Governance, E-Commerce)", "Data breaches", "Data leaks",
    "Attacks on Internet of Things (IoT) devices and associated systems, networks, and servers",
    "Attacks or incidents affecting digital payment systems", "Attacks via malicious mobile apps",
    "Fake mobile apps", "Unauthorized access to social media accounts",
    "Attacks or suspicious activities affecting cloud computing systems, servers, software, and applications",
    "Attacks or malicious/suspicious activities affecting systems related to Big Data, Blockchain, virtual assets, and robotics",
    "Attacks on systems related to Artificial Intelligence (AI) and Machine Learning (ML)", "Backdoor attacks",
    "Disinformation or misinformation campaigns", "Supply chain attacks", "Cyber espionage", "Zero-day exploits",
    "Password attacks", "Web application vulnerabilities", "Hacking", "Malware attacks"
]

# Load subcategory models
financial_fraud_model = BertForSequenceClassification.from_pretrained("Darshankochar022/Financial_Fraud1")
women_child_related_model = BertForSequenceClassification.from_pretrained("Darshankochar022/Women_Child_Crime_Classifier1")
subcategory_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
subcategory_embeddings = embedding_model.encode(other_cyber_crime_subcategories, convert_to_tensor=True)

# Streamlit UI
st.title("CyberCrime Classifier")
st.subheader("Enter Text Directly")
user_text = st.text_area("Type your text here:")

st.subheader("Or Upload a CSV File")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

def classify_category(text):
    inputs = category_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = category_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_to_category.get(prediction, "Unknown Category")

def classify_subcategory(text, category):
    if category == "Financial Fraud Crime":
        model = financial_fraud_model
        subcategory_map = financial_fraud_subcategories
    elif category == "Women & Child Related Crime":
        model = women_child_related_model
        subcategory_map = women_child_related_subcategories
    elif category == "Other Cyber Crime":
        text_embedding = embedding_model.encode(text, convert_to_tensor=True)
        similarities = util.cos_sim(text_embedding, subcategory_embeddings)
        best_match_idx = similarities.argmax().item()
        return other_cyber_crime_subcategories[best_match_idx], similarities[0][best_match_idx].item()
    else:
        return "No subcategory defined for this category."

    inputs = subcategory_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    subcategory_prediction = torch.argmax(outputs.logits, dim=1).item()
    return subcategory_map.get(subcategory_prediction, "Unknown Subcategory")

# Text input processing
if user_text:
    st.write("You entered the following text:")
    st.write(user_text)
    st.markdown("### Processing Text...")

    category = classify_category(user_text)
    st.write(f"Predicted Category: {category}")

    if category != "Unknown Category":
        subcategory = classify_subcategory(user_text, category)
        if isinstance(subcategory, tuple):
            st.write(f"Subcategory: {subcategory[0]}")
            st.write(f"Similarity Score: {subcategory[1]:.2f}")
        else:
            st.write(f"Subcategory: {subcategory}")

# File upload processing
elif uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "crimeaditionalinfo" not in df.columns or "category" not in df.columns or "sub_category" not in df.columns:
        st.error("CSV file must contain 'crimeaditionalinfo', 'category', and 'sub_category' columns.")
    else:
        st.write("CSV File Contents:")
        st.dataframe(df)

        st.markdown("### Processing CSV File...")
        results = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Classifying Rows"):
            text = row["crimeaditionalinfo"]
            category = classify_category(text)
            subcategory = classify_subcategory(text, category) if category != "Unknown Category" else "No Subcategory"
            results.append({"text": text, "category": category, "subcategory": subcategory})

        results_df = pd.DataFrame(results)
        st.write("Final Output from CSV Processing:")
        st.dataframe(results_df)

        # Save output
        results_df.to_csv("output_with_subcategories.csv", index=False)
        st.success("Classification complete. Results saved to output_with_subcategories.csv.")

else:
    st.write("Please enter text or upload a CSV file to proceed.")
