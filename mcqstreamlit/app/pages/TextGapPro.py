import streamlit as st
import textwrap
import json
import requests
import string
import re
import nltk
import string
import itertools

import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from pprint import pprint
import random

st.header(" TextGapPro")
st.subheader("TextGapPro is a cutting-edge Natural Language Processing (NLP) application designed to empower users with the ability to generate fill-in-the-blank sentences effortlessly. In a world increasingly reliant on effective communication, TextGapPro stands out as a powerful tool for writers, educators, and content creators seeking to enhance their content's readability and engagement.")
text = st.text_area("Input the text to get the fill in the blanks",placeholder="Enter the text", height=200)
button = st.button("Generate Fill-in-The-Blank")

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_noun_adj_verb(text):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text,language='en')
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'VERB', 'ADJ', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        # extractor.candidate_selection(pos=pos, stoplist=stoplist)
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=30)


        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

def get_fill_in_the_blanks(sentence_mapping):
    out={"title":"Fill in the blanks for these sentences with matching words at the top"}
    blank_sentences = []
    processed = []
    keys=[]
    for key in sentence_mapping:
        if len(sentence_mapping[key])>0:
            sent = sentence_mapping[key][0]
            # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
            insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
            no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
            line = insensitive_sent.sub(' _________ ', sent)
            if (sentence_mapping[key][0] not in processed) and no_of_replacements<2:
                blank_sentences.append(line)
                processed.append(sentence_mapping[key][0])
                keys.append(key)
    out["sentences"]=blank_sentences[:10]
    out["keys"]=keys[:10]
    return out

if text and button:
    wrapper = textwrap.TextWrapper(width=150)
    word_list = wrapper.wrap(text=text)
    #for element in word_list:
    #print(element)
    #st.write(word_list)
    sentences = tokenize_sentences(text)
    #st.write(sentences)
    noun_verbs_adj = get_noun_adj_verb(text)
    #st.write(noun_verbs_adj)
    keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, sentences)
    #st.write(keyword_sentence_mapping_noun_verbs_adj)
    fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)
    #st.write(fill_in_the_blanks)
    
    # Need to show the shuffle the answer 
    # all_answers = []
    # for keys in fill_in_the_blanks['keys']:
    #     all_answers.append(keys)
    #random.shuffle(all_answers)
    
    # list_answer = list(all_answers)
    # random.shuffle(list_answer)

    st.header("Fill in the blanks from Input")
    # for ans in list_answer:
    #     st.write(ans)
    count =0 
    for sentence in fill_in_the_blanks['sentences']:
        count = count + 1
        st.write(count,sentence)

    st.header(" Correct Answer")
    #st.write(all_answers)
    for key in fill_in_the_blanks['keys']:
        st.write(key)
