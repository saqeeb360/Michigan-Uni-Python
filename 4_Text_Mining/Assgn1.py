# -*- coding: utf-8 -*-
"""
Created on Sun May 24 03:07:49 2020

@author: sony
"""

import re
import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)
df = pd.Series(doc)
df = pd.DataFrame({'text' : df})
df.head(10)

df['dates'] = df['text'].str.extract(r'((?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{1,2}\/\d{4})|(?:(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,? \d{4})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?[a-z]*\.?[ -]\d{2}[snrt]?[tdh]?,?[ -]\d{4})|(?:\d{4}))', expand=False)  
df['dates'][271] = '1 August 2008'
df['dates'][313] = '1 December 1978'
df['dates'][298] = 'January 1993'
df['final'] = pd.to_datetime(df['dates'], errors='coerce')
df.sort_values('final',inplace=True)
df.reset_index(inplace=True)
df['index']
#\d{1,2}[-/]\d{1,2}[-/]\d{2,4}
#\d{1,2}\/\d{4}
#(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,? \d{4}
#(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?[a-z]*\.?[ -]\d{2}[snrt]?[tdh]?,?[ -]\d{4}
#\d{4}

#df[271] = '1 August 2008'

#df['all_three_alpha'] = df['text'].str.extract(r'((?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,? \d{4})')
#df['all_three_alpha_'] = df['text'].str.extract(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?[a-z]*\.?[ -]\d{2}[snrt]?[tdh]?,?[ -]\d{4})')
#df['last'] = df['text'].str.extract(r'(\d{4})')

'''-------------------------------------------------------------'''
doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)
df = pd.DataFrame({'text' : doc})
df = pd.Series(doc)
regex1 = '(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
regex2 = '((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\S]*[+\s]\d{1,2}[,]{0,1}[+\s]\d{4})'
regex3 = '(\d{1,2}[+\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\S]*[+\s]\d{4})'
regex4 = '((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\S]*[+\s]\d{4})'
regex5 = '(\d{1,2}[/-][1|2]\d{3})'
regex6 = '([1|2]\d{3})'
full_regex = '(%s|%s|%s|%s|%s|%s)' %(regex1, regex2, regex3, regex4, regex5, regex6)
parsed_date = df['text'].str.extract(full_regex)
parsed_date = parsed_date.iloc[:,0].str.replace('Janaury', 'January').str.replace('Decemeber', 'December')
parsed_date = pd.DataFrame({0:pd.to_datetime(parsed_date)})
parsed_date['index'] = parsed_date[0].sort_values(ascending=True).index
pd.Series(parsed_date['index'].values)






