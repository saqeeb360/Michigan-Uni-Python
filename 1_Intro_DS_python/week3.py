# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:48:16 2020

@author: sony
"""

import pandas as pd
import numpy as np
df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])

adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
print(staff_df.head())
print()
print(student_df.head())

df = pd.merge(staff_df,student_df, how = 'outer', left_index = True, right_index=True)
df = pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)

df = pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
df = pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)

staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
df = pd.merge(staff_df,student_df,how = 'left', left_on = "Name", right_on = "Name")
df.head()

staff_df["Location"] = [1,2,3]
student_df["Location1"] = ["d","e","f"]
staff_df = staff_df.reset_index()
df = pd.merge(student_df,staff_df,right_on = "Name",left_index= True)

staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
staff_df
student_df
pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])
"""
Pandas Idioms
"""

#Can you use method chaining to modify the DataFrame df in one statement to drop 
#any entries where 'Quantity' is 0 and rename the column 'Weight' to 'Weight (oz.)'?

(df.where(df["Quantity"] == 0)
    .dropna()
    .rename(columns={"Weight":"Weight (oz.)"})

df = pd.read_csv("census.csv")

rows = ["POPESTIMATE2010",
        "POPESTIMATE2011",
        "POPESTIMATE2012",
        "POPESTIMATE2013",
        "POPESTIMATE2014",
        "POPESTIMATE2015"]

df["min"] = df.apply(lambda x : x[rows].min(),axis =1)


"""
Group by
"""
df = pd.read_csv("census.csv")
df = df[df['SUMLEV']==50]

x = df.groupby('STNAME')
for i,j in x: 
    print(i)

%%timeit -n 1
for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])
    print('Counties in state ' + state + ' have an average population of ' + str(avg))    


%%timeit -n 10
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg))

%%timeit -n 10
print(df.groupby('STNAME').agg({'CENSUS2010POP': np.average}))


df = df.set_index('STNAME')
def fun(item):
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2
for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')

df.groupby(fun).agg({'CENSUS2010POP' : len})

print(df.groupby('Category').apply(lambda df,a,b: sum(df[a] * df[b]), 'Weight (oz.)', 'Quantity'))
# Or alternatively without using a lambda:
# def totalweight(df, w, q):
#        return sum(df[w] * df[q])
#        
# print(df.groupby('Category').apply(totalweight, 'Weight (oz.)', 'Quantity'))


print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']))
print(type(df.groupby(level=0)['POPESTIMATE2010']))

(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
    .agg({'avg': np.average, 'sum': np.sum}))

(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP','POPESTIMATE2010']
    .agg({'avg': np.average, 'sum': np.sum}))

"""
Scales
"""
df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df

grades = df['Grades'].astype('category')
grades > "C"

grades = df['Grades'].astype('category',
                             categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)

g_dtype = pd.api.types.CategoricalDtype(categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                                        ordered=True)
grades = df.astype(g_dtype)
grades

grades = df.astype(pd.api.types.CategoricalDtype(categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                                        ordered=True))
grades
grades > "C"

df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})
pd.cut(df['avg'],10)


"""
Pivots
Tables    
"""

df = pd.read_csv('cars.csv')
df
df1 = df.pivot_table(values = ['(kW)','TIME (h)'],  index='YEAR', columns='Make', aggfunc=np.mean)
df1 = df.pivot_table(values = ['(kW)'],  index='YEAR', columns='Make', aggfunc=np.mean)
df1 = df.pivot_table(values = ['(kW)'],  index=['YEAR','Make'])
df1 = df.pivot_table(values = ['(kW)','TIME (h)'],  index=['YEAR','Make'])

"""
Data Functionality
Timestamp, Period, DateTimeIndex - list, PeriodIndex - list, converting datetime, 
timedeltas 
"""

pd.Timestamp('9/1/2016 10:05AM')

pd.Period('2016')

pd.Period('1/2016')

pd.Period('3/5/2016')

t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1

t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2

d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3

ts3.index = pd.to_datetime(ts3.index, dayfirst =True)
ts3

pd.Timestamp('9/3/2021')-pd.Timestamp('9/1/2020')
pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')


dates = pd.date_range('10-10-20',periods=9,freq='2W-SUN')
dates
type(dates)

df = pd.DataFrame({'Count 1':100 + np.random.randint(-5,5,9),'Count 2': 120 + np.random.randint(-5,5,9)}, index = dates)
df
df.index.weekday_name
df.diff()

df.resample('M').mean()
df.resample('Y').mean()
df.resample('D').mean()

df.asfreq('W',method='ffill')

import matplotlib.pyplot as plt
%matplotlib inline
df.plot()










