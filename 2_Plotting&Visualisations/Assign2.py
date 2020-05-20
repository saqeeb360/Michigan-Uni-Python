# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:46:58 2020

@author: sony
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

df = pd.read_excel('course.xlsx', skiprows = 39, usecols = [1,2,3,4], names = ['ID','Date','Element','Data_Value'] )

df['Data_Value'] = df['Data_Value']/10
df['Date'] = pd.to_datetime(df['Date'])

df['Monthname'] = df['Date'].dt.strftime("%B")
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df_2015 = df[df['Date'] >= pd.to_datetime('2015')]
df = df[df['Date'] < pd.to_datetime('2015')]


#df_max = df[df['Element'] == 'TMAX']
#df_min = df[df['Element'] == 'TMIN']
#df_max = df_max.groupby('Date')['Data_Value'].max()
#df_min = df_min.groupby('Date')['Data_Value'].min()
#plt.plot(df_max.index, df_max,'--r')
#plt.plot(df_min.index, df_min,'-o')



#type(df['Date'])
#df.dtypes
#df['Date'] = pd.to_datetime(df['Date'])
#df.dtypes


#df['Date'].dt.year
#df['Date'].dt.day_name
#df['Date'].dt.day_name()
#df['Date'].dt.day
#df['Date'].dt.day.min()
#df['Date'].dt.day.max()
#df['Date'].dt.month_name()




#df['Month'] = df['Date'].dt.strftime("%B")

#df['Month'] = df['Date'].dt.month
#df['Day'] = df['Date'].dt.day
#df['Monthname'] = df['Date'].dt.strftime("%B")

df_max = df[df['Element'] == 'TMAX'].groupby(['Month','Day']).max()
df_min = df[df['Element'] == 'TMIN'].groupby(['Month','Day']).min()
df_max = df_max.reset_index().sort_values(['Month','Day'], ascending= [1,1])
df_min = df_min.reset_index().sort_values(['Month','Day'], ascending= [1,1])


plt.figure()
plt.plot(df_max['Monthname'], df_max['Data_Value'],'--r')
plt.plot(df_min['Monthname'], df_min['Data_Value'],'--b')

my_xticks = list(df_max['Monthname'])
plt.xticks(list(df_max['Month']), my_xticks)
plt.plot(list(df_max['Month']), df_max['Data_Value'])
plt.show()


df_max = df_max[(df_max['Month']!=2) | (df_max['Day'] != 29)]
df_min = df_min[(df_min['Month']!=2) | (df_min['Day'] != 29)]


plt.figure(figsize = [8.4,6.4])
#plt.xticks(list(df_max.index),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'])
plt.xticks(list(range(0, len(df_min),3)), df_min['Monthname'][list(range(0, len(df_min), 3))], rotation = '45')
plt.plot(df_max['Data_Value'])
plt.plot(df_min['Data_Value'])

# fill between
#plt.gca().fill_between(range(len(df_max)) ,
#                       df_max['Data_Value'],
#                       df_min['Data_Value'],
#                       facecolor = 'blue',
#                       alpha = 0.25)
plt.ylabel('Temperature')
plt.legend(['Max 2005-2014','Min 2005-2014','2015 above Max','2015 below Min'] ,loc = 8, frameon = False)
plt.show()



























    
#len(df['Date'].unique())
#min(df['Date']).date()
#max(df['Date']).date()    

#x = df.groupby(['Date','Element'])
#x.groups
#dir(x)
#x.









[['Element']=='TMAX'].max()
x_max = df.groupby(['Date','Element']).agg({'Data_Value': max})

for i, j in x:
    print(i)

x.reset_index(inplace=True)

df_min = x[x['Element']=='TMIN'].sort_values('Date')
df_max = x[x['Element']=='TMAX'].sort_values('Date')


plt.figure()
plt.plot(df_min['Date'], df_min['Data_Value'],'--r',
         df_max['Date'], df_min['Data_Value'], '-o')


