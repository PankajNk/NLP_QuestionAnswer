import nltk
import ntlk_utils #nltk are download in different file
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize

import streamlit as st
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

import time


# st.set_page_config(
#     page_title = "Home",
# )
st.title("NLP Shortcut")

@st.cache_resource
def get_model():
    summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    return summary_model,summary_tokenizer

summary_model,summary_tokenizer = get_model()

input_summary = st.text_area("Input the text to get the summary:",placeholder="Enter the text", height=200) # height in pixel
button = st.button("Press to summarise")

def postprocesstext (content):
  final=""
  for sent in sent_tokenize(content):
    sent = sent.capitalize()
    final = final +" "+sent
  return final

def summarizer(text,model,tokenizer):
  text = text.strip().replace("\n"," ")
  text = "summarize: "+text
  print (text)
  max_len = 512
  encoding = tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=False,truncation=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=3,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  min_length = 75,
                                  max_length=1000)


  dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]
  summary = dec[0]
  summary = postprocesstext(summary)
  summary= summary.strip()

  return summary

if input_summary and button:
    with st.spinner('Please wait...model is processes your input'):
        time.sleep(5)
        summarized_text = summarizer(input_summary,summary_model,summary_tokenizer)
    st.success("Success")
    st.write(summarized_text)
    
    #print("Original:   ",input_summary)
    #print("After :   ",summarized_text)

