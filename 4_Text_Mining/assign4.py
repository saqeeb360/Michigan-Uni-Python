# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 03:02:53 2020

@author: sony
"""
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

doc = 'Fish are nvqjp friends.'
nltk.pos_tag(nltk.word_tokenize('Fish are nvqjp friends.'))
porter = nltk.PorterStemmer()
[porter.stem(t) for t in nltk.word_tokenize('Fish are nvqjp friends.')]

WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in nltk.word_tokenize('Fish are nvqjp friends.')]


tokens = nltk.word_tokenize('Fish are nvqjp friends.')
tokens
pos = nltk.pos_tag(tokens)
pos
tags = [tag[1] for tag in pos]
wntag = [convert_tag(tag) for tag in tags]
ans = list(zip(tokens,wntag))
sets = [wn.synsets(x,y) for x,y in ans]
final = [val[0] for val in sets if len(val) > 0]
final


s=[]
for i1 in s1:
    r=[]
    scores=[x for x in [i1.path_similarity(i2) for i2 in s2]if x is not None]
    if scores:
        s.append(max(scores))
# Your Code Here
sum(s)/len(s)# Your Answer Here





def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None
convert_tag(['1'])

deer = wn.synset('deer.n.01')


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    

    # tokens
    token = nltk.word_tokenize(doc)
    pos = nltk.pos_tag(token)
    doc_pos_tag = [convert_tag(word[1][0]) for word in pos]
    #final_synset = [wn.synsets(x,y)[0] for x,y in zip(token,doc_pos_tag) if len(wn.synsets(x,y)) > 0 ]
    final_synset = list()
    for x,y in zip(token, doc_pos_tag):
        t = wn.synsets(x,y)
        if len(t) > 0:
            final_synset.append(t[0])
            
    return final_synset


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        s1 = doc_to_synsets('I like cats.')
        s2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    # Your Code Here
    s1 = pd.Series(s1)
    s2 = pd.Series(s2)
    def t(word):
        series_max = s2.apply(lambda x : word.path_similarity(x) ).max()    
        return series_max
    final = s1.apply(t).dropna().mean()
    
    return final






def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)


test_document_path_similarity()


















