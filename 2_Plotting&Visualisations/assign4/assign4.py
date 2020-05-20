'''
Population of india from 1951 
http://censusindia.gov.in/Census_Data_2001/India_at_glance/variation.aspx

Religion wise population
http://socialjustice.nic.in/writereaddata/UploadFile/HANDBOOKSocialWelfareStatistice2018.pdf

Kumbh 
https://www.hindustantimes.com/india-news/a-record-over-24-crore-people-visited-kumbh-2019-more-than-total-tourists-in-up-in-2014-17/story-9uncpmhBPnBj11ClnTiYQP.html
https://en.wikipedia.org/wiki/Kumbh_Mela    
https://books.google.com/books?id=9XC9bwMMPcwC&pg=PA242

'''
'''
Islam : 5 basic rules 
1. oath
2. prayers
3. hajj
4. give wealth to poors 
5. Fasting for a month
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('popu.xlsx')
df1 = pd.read_csv('population.csv', sep=' ')
df1 = df1.set_index('ReligiousGroup')
df1 = df1.T
df1.index = df1.index.astype('int64')
total = pd.merge(df,df1,how='inner',left_on='Year',right_index=True)
total.reset_index(drop=True,inplace=True)


def fun(t):
    religion = t[1:]/100
    total = t[0]
    temp = religion*total
    return temp


rel_name = ['Total','Hindu', 'Muslim', 'Christian','Sikh', 'Buddhist', 'Jain', 'Parsi', 'Animist,Others']
total.iloc[:,4:] = total[rel_name].apply(fun,axis=1)
total = total.astype('int')
total.set_index('Year',inplace = True, )

#total[['Total','Hindu','Muslim']].plot()


f = open('hajj.txt','r') 
list1 = f.read().split('\n')   
f.close()
list2 = np.array(list1)
list2.resize(7,7)
list1 = np.array(list1)[49:-1]
list1.resize(25,5)
list1 = list1[:,[0,-1]]
list2 = list2[1:,[0,-1]]
list1 = np.delete(list1,[10],axis=0)
list1[10,0] = 2006
list1 = list1.astype('int')
list2 = list2.astype('int')


hajj = pd.DataFrame(np.concatenate((list2,list1)))
column_name = ['Year', 'Hajj']
hajj.columns = column_name
hajj.set_index('Year',inplace = True)

#hajj.plot()


#z = pd.merge(total,hajj,how='outer',left_index=True,right_index=True)
#z.plot()
'''
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
#plt.yscale('log')
ax.plot(total['Buddhist'] ,'-o',
        total['Parsi'],'-o',
        total['Jain'],'--r')





ax.plot(total['Urban'] ,'-o',
        total['Rural'],'-o',
        total['Total'], '--r',
        total['Buddhist'],'--b')


fig.show()
ax.plot(total.index , total['Muslim'], '-o')
ax.plot(hajj.index , hajj['Hajj'], '-o')
ax.plot(total.index,total['Hindu'],'-o')

'''

experiment = total[['Total','Hindu', 'Muslim', 'Christian','Sikh']]
#experiment[['TotalGw','HinGw','MusGw','ChrGw','SikGw']] = experiment.pct_change(axis='rows').dropna()*100
growth = experiment.pct_change(axis='rows').dropna()*100
hajj['Growth'] = hajj.pct_change()*100

# Kumbha

kumbh = pd.read_excel('kumbh.xlsx')
kumbh['Growth'] = kumbh['Kumbh'].pct_change()*100

#plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('seaborn-ticks')
#from matplotlib.ticker import ScalarFormatter

#%matplotlib notebook

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot()
ax.axis([1950,2020,100,150000000000])
ax.ticklabel_format(useOffset=False)
ax.set_yscale('log')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.plot(experiment.index , experiment['Total'],'-.o',
        experiment.index,experiment['Muslim'],'--',
        experiment.index,experiment['Hindu'],
        experiment.index, experiment['Christian'],'--',
        experiment.index, experiment['Sikh'])
ax.bar(hajj.index-0.2, hajj['Hajj'])
ax.bar(kumbh['Year'],kumbh['Kumbh'])
ax.legend(['Total','Muslim','Hindu','Christian','Sikh','Hajj','Kumbh'],loc=2,
          mode="expand",frameon = False,ncol=4)
ax.set_title('Total Population India Vs Hajj visits Vs Kumbh visits')
ax.set_ylabel('Persons')
#ax.tick_params(top=False, bottom=False, left=True, right=False, labelleft='on', labelbottom='on')
fig.show()


'''

'bmh',
'classic',
'dark_background',
'fast',
'fivethirtyeight',
'ggplot',
'grayscale',
'seaborn-bright',
'seaborn-colorblind',
'seaborn-dark-palette',
'seaborn-dark',
'seaborn-darkgrid',
'seaborn-deep',
'seaborn-muted',
'seaborn-notebook',
'seaborn-paper',
'seaborn-pastel',
'seaborn-poster',
'seaborn-talk',
'seaborn-ticks',
'seaborn-white',
'seaborn-whitegrid',
'seaborn',
'Solarize_Light2',
'tableau-colorblind10'
'_classic_test'

    
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(growth.index,growth['Total'],'-o',
        growth.index,growth['Hindu'],'-o',
        growth.index,growth['Muslim'],'-o',
        growth.index, growth['Christian'],
        growth.index, growth['Sikh'],
        hajj.index, hajj['Growth'],
        kumbh['Year'],kumbh['Growth'])
ax.legend(['Total','Hindu','Muslim','Christian','Sikh','Hajj','Kumbh'])

z = plt.style.available

Indian population Vs Pilgrimage in Hindus and Muslims
'''






