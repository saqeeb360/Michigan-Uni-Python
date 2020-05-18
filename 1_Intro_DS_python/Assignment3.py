# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 03:00:18 2020

@author: sony
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
names = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
'''
energy = pd.read_excel('Energy Indicators.xls',skipfooter=38, encoding='ANSI', header = 9)
energy = energy.iloc[8:,-4:]
energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']

#energy = energy[energy['Energy Supply'] != '...']

'''
def t(x):
    if x == '...':
        return np.NaN
    return x
'''
#energy = energy.copy() 

energy.index = list(range(227))

'''
energy = energy.reset_index()
energy = energy.iloc[:,-4:]
'''
#energy['Energy Supply'] = energy['Energy Supply'].apply(t)
#energy[ 'Energy Supply per Capita'] = energy['Energy Supply per Capita'].apply(t)

energy[ 'Energy Supply'] = energy['Energy Supply'].apply(lambda x :np.NaN if x == '...' else x)
energy['Energy Supply per Capita'] = energy['Energy Supply per Capita'].apply(lambda x :np.NaN if x == '...' else x)


"""

"""
# Conversion of pentajoules to gigajoules
energy['Energy Supply'] = energy['Energy Supply']*1000000

# Changing the name of countrieswith digit

def t(x):
    if x[-2].isdigit():
        return x[:-2]
    if x[-1].isdigit():
        return x[:-1]
    if x[-1] == ")" :
        index = x.index("(")
        return x[:index-1]
    return x
    
energy['Country'] = energy['Country'].apply(t)
    
energy['Country'].replace({ "Republic of Korea" : "South Korea" ,
  "United States of America": "United States",
  "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
  "China, Hong Kong Special Administrative Region": "Hong Kong"},inplace = True)
    
g = energy[energy['Country'].apply(lambda x : False if "".join(x.split()).isalpha() else True)]

"""
Second dataframe
"""
GDP = pd.read_csv("world_bank.csv",header=4)
GDP['Country Name'].replace({"Korea, Rep.": "South Korea",
   "Iran, Islamic Rep.": "Iran",
   "Hong Kong SAR, China": "Hong Kong"},inplace=True)

"""
Third dataframe
"""
ScimEn = pd.read_excel("scimagojr-3.xlsx",header=0)

"""
merge
1. GDP - last 10 years
2. ScimEn top 15
The index of this DataFrame should be the name of the country, and the columns 
should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 
'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', 
'% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
"""
GDP.rename(columns={'Country Name':'Country'},inplace=True)
energy.set_index('Country',inplace=True)
GDP.set_index('Country',inplace=True)
ScimEn.set_index('Country',inplace=True)

df = pd.merge(ScimEn.iloc[:15,:],energy, how = 'inner', left_index=True , right_index=True)
df = pd.merge(df,GDP.iloc[:,-10:], how = 'inner', left_index=True , right_index=True)

"""
ANSWER 1 CODE
def ans_1(): 
    import pandas as pd
    import numpy as np
    energy = pd.read_excel('Energy Indicators.xls',skipfooter=38, encoding='ANSI', header = 9)
    energy = energy.iloc[8:,-4:]
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy.index = list(range(227))
    energy[ 'Energy Supply'] = energy['Energy Supply'].apply(lambda x :np.NaN if x == '...' else x)
    energy['Energy Supply per Capita'] = energy['Energy Supply per Capita'].apply(lambda x :np.NaN if x == '...' else x)
    energy['Energy Supply'] = energy['Energy Supply']*1000000
    def t(x):
        if x[-2].isdigit():
            return x[:-2]
        if x[-1].isdigit():
            return x[:-1]
        if x[-1] == ")" :
            index = x.index("(")
            return x[:index-1]
        return x
    energy['Country'] = energy['Country'].apply(t)
    energy['Country'].replace({ "Republic of Korea" : "South Korea" ,
      "United States of America": "United States",
      "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
      "China, Hong Kong Special Administrative Region": "Hong Kong"},inplace = True)
    energy['% Renewable'] = energy['% Renewable'].astype("float64")
    GDP = pd.read_csv("world_bank.csv",header=4)
    GDP['Country Name'].replace({"Korea, Rep.": "South Korea",
       "Iran, Islamic Rep.": "Iran",
       "Hong Kong SAR, China": "Hong Kong"},inplace=True)
    ScimEn = pd.read_excel("scimagojr-3.xlsx",header=0)
    GDP.rename(columns={'Country Name':'Country'},inplace=True)
    energy.set_index('Country',inplace=True)
    GDP.set_index('Country',inplace=True)
    ScimEn.set_index('Country',inplace=True)
    df = pd.merge(ScimEn.iloc[:15,:],energy, how = 'inner', left_index=True , right_index=True)
    df = pd.merge(df,GDP.iloc[:,-10:], how = 'inner', left_index=True , right_index=True)
    return df

"""

"""
ANSWER  2
"""
def ans_2(): 
    import pandas as pd
    import numpy as np
    energy = pd.read_excel('Energy Indicators.xls',skipfooter=38, encoding='ANSI', header = 9)
    energy = energy.iloc[8:,-4:]
    energy.columns = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy.index = list(range(227))
    energy[ 'Energy Supply'] = energy['Energy Supply'].apply(lambda x :np.NaN if x == '...' else x)
    energy['Energy Supply per Capita'] = energy['Energy Supply per Capita'].apply(lambda x :np.NaN if x == '...' else x)
    energy['Energy Supply'] = energy['Energy Supply']*1000000
    def t(x):
        if x[-2].isdigit():
            return x[:-2]
        if x[-1].isdigit():
            return x[:-1]
        if x[-1] == ")" :
            index = x.index("(")
            return x[:index-1]
        return x
    energy['Country'] = energy['Country'].apply(t)
    energy['Country'].replace({ "Republic of Korea" : "South Korea" ,
      "United States of America": "United States",
      "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
      "China, Hong Kong Special Administrative Region": "Hong Kong"},inplace = True)
    energy['% Renewable'] = energy['% Renewable'].astype("float64")
    GDP = pd.read_csv("world_bank.csv",header=4)
    GDP['Country Name'].replace({"Korea, Rep.": "South Korea",
       "Iran, Islamic Rep.": "Iran",
       "Hong Kong SAR, China": "Hong Kong"},inplace=True)
    ScimEn = pd.read_excel("scimagojr-3.xlsx",header=0)
    GDP.rename(columns={'Country Name':'Country'},inplace=True)
    energy.set_index('Country',inplace=True)
    GDP.set_index('Country',inplace=True)
    ScimEn.set_index('Country',inplace=True)
    
    df = pd.merge(ScimEn,energy, how = 'inner', left_index=True , right_index=True)
    df = pd.merge(df,GDP.iloc[:,-10:], how = 'inner', left_index=True , right_index=True)

    df1 = pd.merge(ScimEn,energy, how = 'outer', left_index=True , right_index=True)
    df1 = pd.merge(df1,GDP.iloc[:,-10:], how = 'outer', left_index=True , right_index=True)
    
    answer_2 = len(df1.index)-len(df.index)
    return answer_2
print(ans_2())
"""
ANSWER 3
"""
import numpy as np
df = Top15
df = df[:15]
avgGDP = df[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014','2015']]
avgGDP = df.mean(axis=1)
avgGDP.sort_values(ascending=False ,inplace=True)


#t = df[['2006','2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean(axis=1).rename('aveGDP').sort_values(ascending=False)

"""
ANSWER 4
"""
answer_4 = df.loc['United Kingdom']['2015'] - df.loc['United Kingdom']['2006']


"""
ANSWER 5
"""

answer_5 = df['Energy Supply per Capita'].mean()


"""
ANSWER 6
"""

x = df[df['% Renewable'] == df['% Renewable'].max()]
answer_6 =(x.index[0],x['% Renewable'][0])

"""
ANSWER 7
"""

df['Citation ratio'] = df['Self-citations']/df['Citations']
x = df['Citation ratio'].max()
t = df[df['Citation ratio']== x]
answer_7 = (t.index[0],x)

"""
ANSWER 8
"""

df['PopEst'] = df['Energy Supply']/df['Energy Supply per Capita']
answer_8 = df['PopEst'].sort_values(ascending = False).index[2]

"""
ANSWER 9 
"""

df['PopEst'] = df['Energy Supply']/df['Energy Supply per Capita']
df['Citable doc per person'] = df['Citable documents']/df['PopEst']
x = df.corr(method = 'pearson')
answer_9 = x.loc['Citable doc per person','Energy Supply per Capita']

"""
import matplotlib as plt
%matplotlib inline 
Top15 = df
Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])
"""

"""
ANSWER 10
"""

x = df['% Renewable'].median()
df['HighRenew'] = [1 if i >=x else 0 for i in df['% Renewable']]
answer_10 = df['HighRenew'].sort_values(ascending=True)

"""
ANSWER 11
"""


import pandas as pd
import numpy as np
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
df = ans_1()
df['PopEst'] = (df['Energy Supply'] / df['Energy Supply per Capita']).astype(float)
df = df.reset_index()
df['Continent'] = [ContinentDict[country] for country in df['Country']]
#answer_11 = df.set_index('Continent').groupby(level=0)['PopEst'].agg({'size': np.size, 'sum': np.sum, 'mean': np.mean,'std': np.std})
answer_11 = df.set_index('Continent').groupby(level=0)['PopEst'].agg(size = np.size, sum = np.sum, mean = np.mean, std = np.std)
answer_11 = answer_11[['size', 'sum', 'mean', 'std']]
return answer_11


'''
df['Continent'] = list(ContinentDict.values())
df = df.reset_index()
df.set_index('Continent',inplace=True)
df['PopEst'] = df['Energy Supply']/df['Energy Supply per Capita']
answer_11 = df.groupby(level=0)['PopEst'].agg({'size':np.size,'sum':np.sum,'mean':np.mean,'std':np.std})
answer_11 = answer_11[['size', 'sum', 'mean', 'std']]
'''
"""
Top15 = df
ContinentDict  = {'China':'Asia',
                  'United States':'North America', 
                  'Japan':'Asia',
                  'United Kingdom':'Europe',
                  'Russian Federation':'Europe',
                  'Canada':'North America',
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
Top15 = Top15.reset_index()
Top15['Continent'] = Top15['Country'].map(ContinentDict)
Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
result = Top15.copy()
result = result[['Continent', 'PopEst']]
result = result.groupby('Continent')['PopEst'].agg({'size': np.size,'sum': np.sum,'mean': np.mean,'std': np.std})
#result = grouped.agg(['np.size', 'sum', 'mean', 'std'])
idx = pd.IndexSlice
#result = result.loc[:, idx['PopEst']]
#result = result.reset_index()
#result = result.set_index('Continent')
"""



"""
ANSWER 12
"""
import pandas as pd
import numpy as np
df = ans_1()
ContinentDict  = {'China':'Asia', 
              'United States':'North America', 
              'Japan':'Asia', 
              'United Kingdom':'Europe', 
              'Russian Federation':'Europe', 
              'Canada':'North America', 
              'Germany':'Europe', 
              'India':'Asia',
              'France':'Europe', 
              'South Korea':'Asia', 
              'Italy':'Europe', 
              'Spain':'Europe', 
              'Iran':'Asia',
              'Australia':'Australia', 
              'Brazil':'South America'}
df = df.reset_index()
df['Continent'] = [ContinentDict[country] for country in df['Country']]
df['bins'] = pd.cut(df['% Renewable'],5)
x = df.groupby(['Continent','bins']).size()
x
"""
import pandas as pd
import numpy as np
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
df['PopEst'] = df['Energy Supply']/df['Energy Supply per Capita']
df['Continent'] = ContinentDict.values()
df['% Renewable'] = pd.cut(df['% Renewable'],5)
df.reset_index(inplace=True)
df.set_index(['Continent','% Renewable'],inplace=True)
x = df.groupby(level=[0,1])['Rank'].agg(np.size)
x.dropna(inplace=True)
"""

"""
ANSWER_13
"""
df['PopEst'] = df['Energy Supply']/df['Energy Supply per Capita']
df['PopEst'] = ['{:,}'.format(i) for i in df['PopEst']]
answer_13 = df['PopEst']


"""
OPTIONAL
"""
import matplotlib as plt
%matplotlib inline
Top15 = df
ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                   '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

for i, txt in enumerate(Top15.index):
    ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

print("This is an example of a visualization that can be created to help understand the data. \
This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' \
2014 GDP, and the color corresponds to the continent.")

























































plt.plot(gdp.columns[5:],gdp.iloc[106,5:],color='green')
plt.plot(gdp.columns[5:],gdp.iloc[38,5:],color='red')
plt.plot(gdp.columns[5:],gdp.iloc[248,5:],color='blue')
plt.show()









any(np.isnan(energy['% Renewable']))

np.isnan(pd.Series([1,2]))

all(energy['% Renewable'].apply(lambda x : ((type(x) == float ) or (type(x) == int))))





