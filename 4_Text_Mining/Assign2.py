# -*- coding: utf-8 -*-
"""
Created on Sun May 31 12:05:26 2020

@author: sony
"""

import nltk
import pandas as pd
import numpy as np
from nltk.book import FreqDist



# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


len(nltk.word_tokenize(moby_raw)) # or alternatively 
len(text1)
text1[:20]

len(set(nltk.word_tokenize(moby_raw))) # or alternatively 
len(set(text1))

lemmatizer = nltk.WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]
len(set(lemmatized))

# ANSWER1
ratio = len(set(text1))/len(text1)


#ANSWER2
freq = FreqDist(text1)
freq['whale']
freq['Whale']

#ANSWER3
freq = FreqDist(text1)
freq.most_common(n=20)

#ANSWER4 

freq = FreqDist(text1)
freq_150 = sorted([key for key in freq if len(key) > 5 and freq[key] > 150] )
freq_150
#ANSWER 5

words = list(set(text1))
longest = ''
for word in words:
    if len(word) > len(longest):
        longest = word
(longest,len(longest))


#ANSWER6
freq = FreqDist(text1)
freq_2000 = sorted([(freq[word],word) for word in freq.keys() if freq[word] > 2000 and word.isalpha()], reverse = True)
freq_2000

#ANSWER7

sents = nltk.sent_tokenize(moby_raw)
ratio = len(text1)/len(sents)
ratio

#ANSWER8

tags = nltk.pos_tag(text1)
tag_dict = dict()

for tag in tags:
    if tag[1] not in tag_dict.keys():
        tag_dict[tag[1]] = 1
    else:
        tag_dict[tag[1]] = tag_dict[tag[1]] + 1
tag_dict
sort_list = sorted(tag_dict.items(), key = lambda x : x[1], reverse = True)
sort_list[:5]

'''
PART2
'''

#ANSWER9

from nltk.corpus import words

correct_spellings = words.words()

entries=['cormulent', 'incendenece', 'validrate']


final_words = list()

for word in entries:
    recom_dict = dict()
    start = word[0]
    end = chr(ord(word[0])+1)
    start_index = correct_spellings.index(start)
    end_index = correct_spellings.index(end)
    for recom_word in correct_spellings[start_index:end_index+1]:
        word_set = set(word)
        recom_dict[recom_word] = nltk.jaccard_distance(word_set,set(recom_word))
    recom_dict = sorted(recom_dict.items(), key = lambda x : x[1] , reverse=False)
    final_words.append(recom_dict[0][0])
        
final_words

#ANSWER10

entries=['cormulent', 'incendenece', 'validrate']
final_words = list()
for word in entries:
    recom_dict = dict()
    start = word[0]
    end = chr(ord(word[0])+1)
    start_index = correct_spellings.index(start)
    end_index = correct_spellings.index(end)
    for recom_word in correct_spellings[start_index:end_index+1]:
        word_ng = set(nltk.ngrams(word, n=4))
        recom_ng = set(nltk.ngrams(recom_word, n=4))
        recom_dict[recom_word] = nltk.jaccard_distance(word_ng,recom_ng)
    recom_dict = sorted(recom_dict.items(), key = lambda x : x[1] , reverse=False)
    final_words.append(recom_dict[0][0])
        
final_words

#ANSWER11


entries=['cormulent', 'incendenece', 'validrate']
final_words = list()
for word in entries:
    recom_dict = dict()
    start = word[0]
    end = chr(ord(word[0])+1)
    start_index = correct_spellings.index(start)
    end_index = correct_spellings.index(end)
    for recom_word in correct_spellings[start_index:end_index+1]:
        recom_dict[recom_word] = nltk.edit_distance(word,recom_word)
    recom_dict = sorted(recom_dict.items(), key = lambda x : x[1] , reverse=False)
    final_words.append(recom_dict[0][0])
        
final_words




































