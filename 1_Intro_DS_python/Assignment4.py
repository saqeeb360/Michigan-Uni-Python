# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:10:42 2020

@author: sony
"""
'''
ANSWER 1
'''
# % %timeit -n 1

import pandas as pd
import numpy as np

def get_list_of_university_towns():
    with open("university_towns.txt",mode = "r", encoding="utf-8") as x:
        t = x.readlines()
    
    '''
    dict1['A']= [] 
    dict1['A'].append(5)
    '''
    
    dict1 = dict()
    
    for i in t:
        i = str(i)
        index = i.find('[edit]')
        if i.find('[edit]') > -1:
            j = i[:index]
            dict1[j] = []
        else:
            index = i.find(' (')
            dict1[j].append(i[:index])
            
    df2 = pd.DataFrame(columns=["State","RegionName"])
    for i in dict1.keys():
        for j in dict1[i]:
            df2 = df2.append({"State":i,"RegionName":j}, ignore_index=True)
    return df2
df1 = get_list_of_university_towns()
'''
df['A'] = list(range(len(df)))
df.set_index(['State','RegionName'],inplace = True)
df.head()
'''
'''
ANSWER 2 & 3 & 4
'''
def recession():
    import pandas as pd
    import numpy as np
    
    df = pd.read_excel('gdplev.xls',skiprows=220,header=None,usecols=[4,6],names=['Quarterly','GDP in billions'])
    list_srt = []
    list_end = []
    
    flag = False
    length = len(df)
    for i in range(length-1):
        if flag==False and (df.iloc[i-1,1] > df.iloc[i,1] > df.iloc[i+1,1]):
            flag = True
            list_srt.append(i-1)
        elif flag== True and (df.iloc[i-1,1] < df.iloc[i,1] < df.iloc[i+1,1]):
            flag = False
            list_end.append(i+1)
    start = df['Quarterly'][list_srt[0]]
    end =  df['Quarterly'][list_end[0]]
    bottom = df['GDP in billions'][list_srt[0]:list_end[0]+1].min()
    bottom = df[df['GDP in billions'] == bottom].reset_index()['Quarterly'][0]
    return start,end,bottom
start,end,bottom = recession()
"""
ANSWER 5
"""
def convert_housing_data_to_quarters():
    import pandas as pd
    import numpy as np
    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming',
              'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah',
              'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee',
              'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas',
              'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan',
              'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi',
              'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota',
              'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut',
              'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas',
              'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California',
              'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico',
              'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire',
              'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}
    
    df = pd.read_csv('City_Zhvi_AllHomes.csv')
    quar_list = []
    count = 1
    year = 2000
    for i in range(67):
        if count == 1 :
            quar_list.append(str(year)+'q'+str(count))
            count+=1
        elif count == 2:
            quar_list.append(str(year)+'q'+str(count))
            count+=1
        elif count == 3:
            quar_list.append(str(year)+'q'+str(count))
            count+=1
        else:
            quar_list.append(str(year)+'q'+str(count))
            count = 1
            year +=1
    #df1 = pd.read_excel('gdplev.xls',skiprows=220,header=None,usecols=[4,6],names=['Quarterly','GDP in billions'])
    #quar_list = list(df1['Quarterly'])
    #quar_list.append('2016q3')
    col = [1,2]
    col.extend(list(range(51,251)))
    t = df.iloc[:,col].copy()
    t.loc[:,'2016-09'] = 0
    year = list(t.columns) 
    year = year[2:]    
    length = len(year)
    new_df = df.iloc[:,1:3]
    for i in range(1,length,3):
        new_df[quar_list[(i-1)//3]] = (t[year[i-1]]+t[year[i]]+t[year[i+1]])/3

    new_df['State'].replace(states,inplace = True)
    new_df.set_index(['State','RegionName'],inplace=True)
    new_df = new_df.sort_index()   
    return new_df

df = convert_housing_data_to_quarters()
    
    
"""
ANSWER 5
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

df1 = get_list_of_university_towns()
df = convert_housing_data_to_quarters()

#start
#bottom

t = df.copy()
t.reset_index(inplace=True)
s_index = t.columns.get_loc(start)
s_index-=1
t['recession_diff'] = t.iloc[:,s_index] / t[bottom]

#t['2008q2'] , t['2009q2']


x_uni = pd.merge( t[['State','RegionName','2008q1','2009q2','recession_diff']],df1, how='inner', on=['State','RegionName'])

#x_uni['c'] = True
#x_uni.dropna(axis=0,inplace=True)
#y_notuni = pd.merge(t,df1 ,how='left', left_on=['State','RegionName'], right_on =['State','RegionName'])
#y_notuni.dropna(axis=0,inplace=True)

y_notuni = t[['State','RegionName','2008q1','2009q2','recession_diff']].merge(df1, indicator='i', how='outer').query('i == "left_only"').drop('i', 1)

s, p = ttest_ind(y_notuni['recession_diff'],x_uni['recession_diff'],nan_policy= 'omit')
x_uni['recession_diff'].mean()
y_notuni['recession_diff'].mean()

better = "university town"
return (True,p,better)



plt.hist(y['2009q2'],bins=100,color='red',label='y')
plt.hist(x['2009q2'],bins=100,color='blue',label='x')
plt.legend(loc='upper right')
plt.show()
plt.plot(list(range(1,10462)),y['2009q2'].sort_values())
plt.plot(list(range(1,262)),x['2009q2'].sort_values())




'''















dates = pd.period_range('2000-01','2016-9', freq='q')

"""
2000 1 Quater
"""
