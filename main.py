import sys
import pandas as pd
from sklearn.metrics import f1_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification

category_models = {
    "category1": "path/to/your/category1_model",
    "category2": "path/to/your/category2_model",
    "category3": "path/to/your/category3_model",
}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load('category2.bin'))
model = model.to(device)

def predict(text, model, tokenizer, device):
    model.eval()

    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, prediction = torch.max(logits, dim=1)
    
    return prediction.item()

text = "Dear Sir Please stop the fraudulent transaction and refund the amount in source accountDebit Card N Regards"

prediction = predict(text, model, tokenizer, device)
if(prediction == 0):
    print(f"Category :"+ "financial fraud crimes")
elif(prediction == 1):
    print(f"Category :"+"Other Cyber Crimes")
else:
    print(f"Category :"+"Women /Child related crimes")    

