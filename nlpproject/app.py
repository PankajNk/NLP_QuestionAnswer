import streamlit as st
import numpy as np 
from transformers import BertTokenizer, BertForSequenceClassification
import torch

#@st.cache(allow_output_mutation=True)
@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("PankajNk/toxichub")
    return tokenizer,model

tokenizer,model = get_model()

user_input = st.text_area('Enter Test to be Analyze')
button = st.button("Analyze")


d ={
    1:'Toxic',
    0:'Non Toxic'
}


if user_input and button:
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    outputs = model(**test_sample)
    #predication = torch.nn.functional.softmax(outputs.logits, dim = 1)
    st.write("logits: ", outputs.logits)
    y_predication = np.argmax(outputs.logits.detach().numpy(), axis =1)
    st.write("Predication",d[y_predication[0]])