# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:06:14 2020

@author: saqeeb
"""

"""
Pandas Series data structure
"""

import pandas as pd
import numpy as np
x = pd.Series([1,2,3,4],index = ['a','b','c','d'])
x
animal = ['Tiger', 'Lion' , 'Moose' ,None]
pd.Series(animal)

numbers = [1,2,3,None]
pd.Series(numbers)

None == np.NaN # nan - not a number
np.NaN == np.NaN
np.isnan(np.NaN)

sports = {'Archery' : 'Bhutan',
          'Golf' : 'Scotland',
          'Sumo' : 'Japan',
          'Teakwood': 'South Korea'}
s = pd.Series(sports)
s
"""
Querying a Series
"""

s.iloc[3]
s.loc['Golf']

x = {1:'a',
     2:'b',
     3:'c'}

x = pd.Series(x)
x
x[0] #error
x.iloc[0]

s = pd.Series(np.random.randint(0,1000,10000))
total = 0

for item in s:
    total += item
total

np.sum(s)

%%timeit -n 100
summary = 0
for item in s:
    summary += item

%%timeit -n 100
summary = np.sum(s)

%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,1000))
for label,value in s.iteritems():
    s.loc[label] = value+2

%%timeit -n 10
s = pd.Series(np.random.randint(0,1000,1000))
s+=2


"""
Dataframe Data structure
"""

p_1 = pd.Series({'name' : 'saqeeb',
                 'item':'laptop',
                 'cost' : 1000})

p_2 = pd.Series({'name' : 'shaista',
                 'item':'tv',
                 'cost' : 2000})

p_3 = pd.Series({'name' : 'shaheen',
                 'item':'fridge',
                 'cost' : 3000})

df = pd.DataFrame([p_1,p_2,p_3],index = ['s1','s1','s3'])
df
df.loc['s3']
df.loc['s1']
df.iloc[1]
df.iloc[:]

type(df.loc['s3'])
type(df.loc['s1'])
type(df.iloc[1])
type(df.iloc[:])


df.loc['name']
df.loc['s1']
df['name']

df = pd.DataFrame([p_1,p_2,p_3],index = ['s1','s1','cost'])
df.loc['cost']
df
df.loc[:,['cost']]

df.drop('cost') #index(0,1,2) or label(s1,s2,s3)
df
df_copy = df.copy()
df_copy
del df_copy['item']
df_copy

df_copy = df.copy()
del df_copy['cost']
df_copy

df['location'] = pd.Series([1,2,3])
df

df['location'] = pd.Series([1,2,3],index = ['s1','s1','cost'])
df

# 20% discount to the cost
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
#df.loc[:,['Cost']] = df.loc[:,['Cost']]*.8
df['Cost'] *= .8

"""
Data indexing and loading
"""
file1 = open('olympics.csv')
file1.readlines()
file1.close()
flie1

df = pd.read_csv?('olympics.csv')
df.head(5)

df = pd.read_csv("olympics.csv" , skiprows=1)
df.head(5)

df = pd.read_csv("olympics.csv" ,encoding='UTF8', skiprows=1, index_col = 0, skipfooter=1)
df.head(5)

#renaming the columns 
df.columns
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col : 'Gold'+col[4:]}, inplace = True)
    if col[:2]=='02':
        df.rename(columns={col : 'Silver'+col[4:]}, inplace = True)
    if col[:2]=='03':
        df.rename(columns={col : 'Bronze'+col[4:]}, inplace = True)
    if col[0]=='№':
        df.rename(columns={col : '#'+col[2:]}, inplace = True)


"""
Querying a Dataframe
"""

only_gold = df.where(df['Gold'] > 0)
only_gold

only_gold['Gold'].count()
df['Gold'].count()

only_gold = only_gold.dropna()


only_gold = df[df['Gold'] > 0]

%%timeit -n 100
for i in range(1000):
    x = True | False

%%timeit -n 100
for i in range(1000):
    x = True or False

only_gold = df [ (df['Gold'] > 0 ) & (df['Gold.1'] > 0) ]

only_gold = df.where((df['Gold'] > 0) &(df['Gold.1'] > 0 ))

"""
Indexing Dtaframes
"""

df = pd.read_csv('census.csv')
df.head(5)

df['SUMLEV'].unique()
df = df[df['SUMLEV'] == 50]
list1 = df.columns
columns_to_keep = list(list1[[5,6,21,22,23,24,25,26,9,10,11,12,13,14]])

df = df[columns_to_keep]
df = df.set_index(['STNAME','CTYNAME'])
df.head(5)

df.loc['Alabama','Autauga County']
df.loc[[('Alabama','Autauga County'),('Alabama','Baldwin County')]]


purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df['Location'] = df.index
df = df.set_index(['Location' , 'Name'])
df.head(5)

df.append(pd.Series({'Item Purchased' : 'Food',
                     'Cost' : 5.0}, name = ('Store 2','Saqeeb')))


"""
Missing Values 
"""

df = pd.read_csv('log.csv')
df.head(5)

df = df.set_index(['time','user'])
df.head(5)


"""
Assignment
"""

# Max summer gold country name
'''df = pd.read_csv("olympics.csv" ,encoding='UTF8', skiprows=1, engine = 'python',index_col = 0, skipfooter=1)
for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col : 'Gold'+col[4:]}, inplace = True)
    if col[:2]=='02':
        df.rename(columns={col : 'Silver'+col[4:]}, inplace = True)
    if col[:2]=='03':
        df.rename(columns={col : 'Bronze'+col[4:]}, inplace = True)
    if col[0]=='№':
        df.rename(columns={col : '#'+col[2:]}, inplace = True)
'''
df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')

df[df['Gold'] == df['Gold'].max()].index[0]

# Country having maximum difference between summer and winter medal count
df[df['Total']-df['Total.1'] == (df['Total']-df['Total.1']).max() ].index[0]

# biggest difference between their summer gold medal counts and winter 
# gold medal counts relative to their total gold medal count
df[(df['Gold']>0 )&(df['Gold.1']>0)][((df['Gold']-df['Gold.1'])/df['Gold.2']) == ((df['Gold'] - df['Gold.1']) / df['Gold.2']).max()]
df = df[(df['Gold']>0 )&(df['Gold.1']>0)]
df = df[((df['Gold']-df['Gold.1']).abs()/df['Gold.2']) == ((df['Gold'] - df['Gold.1']).abs() / df['Gold.2']).max()]
df.index[0]
# Points - Gold = 3, Silver = 2, Bronze = 1
len((df['Gold.2']*3 ) + (df['Silver.2']*2) + (df['Bronze.2']))

#census data

#state with most counties
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]

df[['STNAME','CTYNAME']].describe()
df['STNAME'].value_counts().index[0]

#ANSWER 6
cdf = census_df[census_df['SUMLEV'] == 50]
cdf = cdf.groupby('STNAME')
cdf = cdf.apply(lambda x:x.sort_values('CENSUS2010POP', ascending=False)).reset_index(drop=True)
cdf = cdf.groupby('STNAME').head(3)
cdf = cdf.groupby('STNAME').sum()
cdf = cdf.sort_values('CENSUS2010POP', axis=0, ascending=False).head(3)
list(cdf.index)

#Answer 7
df = df[df['SUMLEV']==50]
df = df.set_index(['CTYNAME'])
df = df.iloc[:,8:14]
df[(df.max(axis=1)-df.min(axis =1))==((df.max(axis=1)-df.min(axis =1)).max())].index[0]

#Answer 8
df = df[df['SUMLEV']==50]
df = df[(df['REGION']<3) & (df['POPESTIMATE2015']>df['POPESTIMATE2014'])]
a = df['CTYNAME'].apply(lambda x: x if x[:10]=='Washington' else 1)

df['CTYNAME'] = a
df = df[df['CTYNAME'].apply(lambda x: str(x)[0].isalpha())]
df = df[['STNAME','CTYNAME']]
df



