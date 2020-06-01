# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:41:42 2020

@author: sony
"""

import nltk

from nltk.book import *

len(text2)

text7
sent7
len(sent7)
len(text7)

len(set(text7))
list(set(text7))[:10]

dist = FreqDist(text7)

vocab1 = list(dist.keys())
vocab1[:9]
freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]

input1 = 'list listed lists listing listings'
words1 = input1.lower().split()

porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]

udhr = nltk.corpus.udhr.words('English-Latin1')
type(udhr)

udhr[:20]
[porter.stem(t) for t in udhr[:20]]

WNlemma = nltk.WordNetLemmatizer()
[WNlemma.lemmatize(t) for t in udhr[:20]]

text11 = "Children shouldn't drink a sugary drink before bed."
text11.split()
nltk.word_tokenize(text11)

text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!" 
nltk.sent_tokenize(text12)


# PARTS OF SPEECH TAGGING

nltk.help.upenn_tagset('RB')

text13 = nltk.word_tokenize(text11)
nltk.pos_tag(text13)

text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
nltk.pos_tag(text14)

text15 = nltk.word_tokenize("Alice loves Bob")
nltk.pos_tag(text15)
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

grammar
print(grammar)
parser = nltk.ChartParser(grammar)
type(parser)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


from nltk.corpus import treebank
print(treebank.parsed_sents('wsj_0001.mrg')[0])





