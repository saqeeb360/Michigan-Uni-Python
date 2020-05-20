# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:52:23 2020

@author: sony
"""
%matplotlib notebook
%matplotlib inline

import matplotlib as mpl
mpl.get_backend()

import matplotlib.pyplot as plt

plt.plot?

plt.plot(3,2,'.')

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

fig = Figure()
canvas = FigureCanvasAgg(fig)
ax = fig.add_subplot(111)  # 111 mean we just want one plot no subplot
ax.plot(3,2,'.')
canvas.print_png('test.png')
%%html
<img scr='test.png' />


plt.figure()
plt.plot(3,2,'o')
ax = plt.gca()
ax.axis([0,6,0,10])

plt.figure()
plt.plot(1.5,1.5,'o')
plt.plot(2.5,2.5,'o')
plt.plot(2,2,'o')
plt.plot(1.5,4.5,'o')

plt.figure()
plt.plot(1.5,1.5,'o')
plt.plot(2.5,2.5,'o')
plt.plot(2,2,'o')
plt.plot(1.5,4.5,'o')
ax = plt.gca()
ax.get_children()

# Scatter plot

plt.plot([1,2,3,4],[4,3,2,1])
plt.scatter([1,2,3,4],[1,2,3,4])

import numpy as np

x = np.array([1,2,3,4])
y = x

plt.figure()
plt.scatter(x,y)

x = np.array([1,2,3,4])
y = x
colours = ['green']*(len(x)-1)
colours.append('red')
plt.figure()
plt.scatter(x,y, s = 100, c = colours)

'''
zip_gen = zip([1,2,3,4], [5,6,7,8])
list(zip_gen)

zip_gen = zip([1,2,3,4], [5,6,7,8])
x,y = zip(*zip_gen)
x
y
'''

plt.figure()
plt.scatter(x[:2],y[:2], s =300,c='red',label = 'Tall student')
plt.scatter(x[2:],y[2:], s =100,c='blue',label = 'Short student')
plt.legend()

plt.figure()
plt.scatter(x[:2],y[:2], s =300,c='red',label = 'Tall student')
plt.scatter(x[2:],y[2:], s =100,c='blue',label = 'Short student')
plt.legend(loc=4, frameon = False,title = 'Legend')

# Line Plots


linear_data = np.array([1,2,3,4,5,6,7,8])
quad_data = linear_data**2

plt.figure()
plt.plot(linear_data,'-o',quad_data,'-o')
plt.plot([22,44,55],'--b') # '--b' : dash with blue color
plt.xlabel("some data")
plt.ylabel("some more data")
plt.title("A Title")
plt.legend(['Baseline','Competition','US'],loc=2,frameon=True)
plt.gca().fill_between(range(len(linear_data)) ,
                        linear_data,quad_data,
                        facecolor = 'blue',
                        alpha = 0.25)


plt.figure()
plt.plot(linear_data,'-o',quad_data,'-o')
plt.plot([22,44,55],'--b') # '--b' : dash with blue color
plt.xlabel("some data")
plt.ylabel("some more data")
plt.title("A Title")
plt.legend(['Baseline','Competition','US'],loc=2,frameon=True)
plt.gca().fill_between(range(2,5) ,
                        linear_data[2:5],quad_data[2:5],
                        facecolor = 'blue',
                        alpha = 0.25)

plt.figure()
ob_dates = np.arange('2017-01-01','2017-01-09',1,dtype= 'datetime64[D]')
plt.plot(ob_dates,linear_data,'-o',
         ob_dates,quad_data,'-o')

import pandas as pd
plt.figure()
ob_dates = np.arange('2017-01-01','2017-01-09',2,dtype = 'datetime64[D]')
ob_dates = map(pd.to_datetime,ob_dates) #error as map cannot be taken - convert to list
ob_dates
plt.plot(ob_dates,linear_data)

plt.figure()
ob_dates = np.arange('2017-01-01','2017-01-09',1,dtype = 'datetime64[D]')
ob_dates = map(pd.to_datetime,ob_dates) 
ob_dates = list(ob_dates)
plt.plot(ob_dates,linear_data,'-o',
         ob_dates,quad_data,'-o')#error as map cannot be taken - convert to list
x = plt.gca().xaxis
for item in x.get_ticklabels():
    item.set_rotation(45)
plt.subplots_adjust(bottom=0.10)
ax = plt.gca()
ax.set_xlabel('Date')
ax.set_ylabel('Units')
ax.set_title('Quadratic ($x^2$) vs. Linear ($x$) performance')

# Bar charts 

linear_data = np.array([1,2,3,4,5,6,7,8])
quad_data = linear_data**2

plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals,linear_data,width=0.3)
new_xvals = []
for item in xvals:
    new_xvals.append(item+0.3)
plt.bar(new_xvals,quad_data,width=0.3,color = 'red')

from random import randint
plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals,linear_data,width=0.3)
new_xvals = []
for item in xvals:
    new_xvals.append(item+0.3)
plt.bar(new_xvals,quad_data,width=0.3,color = 'red')
linear_err = [randint(0,15) for i in linear_data]
plt.bar(xvals,linear_data,width=0.3, yerr = linear_err)


plt.figure()
plt.bar(xvals,linear_data,width=0.3, color='b')
plt.bar(xvals,quad_data,width=0.3, color='r', bottom= linear_data)

plt.figure()
plt.barh(xvals,linear_data,height=0.3,color='b')
plt.barh(xvals,quad_data,height=0.3,color='r',left = linear_data)


# Dejunkify charts


languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

plt.figure()
plt.bar(pos, popularity, align='center')
plt.xticks(pos, languages)
plt.ylabel('% Popularity')
plt.xlabel("hello")
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)
plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=True,
                labelright=False, labeltop = False,
                labelrotation = 45)
x = plt.gca().xaxis
for spine in plt.gca().spines.values():
    spine.set_visible(False)
x.get_children()

#TODO: remove all the ticks (both axes), and tick labels on the Y axis
#plt.xticks([],[])
#plt.yticks([],[])
#plt.tick_params(axis = 'y',
#                bottom= False,
#                top=False,
#                labelbottom = False)
#plt.axis('off')

dir(plt.gca().spines.values().__class__)
x = plt.gca().spines.values()
for spine in x:
    print(dir(spine))
    break






import matplotlib.pyplot as plt
import numpy as np

plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

# change the bar color to be less bright blue
bars = plt.bar(pos, popularity, align='center', linewidth=0, color='lightslategrey')
# make one bar, the python bar, a contrasting color
bars[0].set_color('#1F77B4')

# soften all labels by turning grey
plt.xticks(pos, languages, alpha=0.8)
# remove the Y label since bars are directly labeled
#plt.ylabel('% Popularity', alpha=0.8)
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha=0.8)

# remove all the ticks (both axes), and tick labels on the Y axis
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# remove the frame of the chart
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# direct label each bar with Y axis values
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 
                 ha='center', color='w', fontsize=11)
plt.show()

dir(plt.gca().
plt.gca?
dir(matplotlib.axes.Axes.text)
matplotlib.axes.Axes?
matplotlib.text.Text?


dir(plt.gca().text())
dir(plt.gca().text)
dir(plt.gca().text())
plt.gca?
matplotlib.axes.Axes
dir(matplotlib.axes.Axes)
dir(matplotlib.axes.Axes.text)
dir(matplotlib.axes.Axes.text())
plt.gca?
dir(matplotlib.axes.Axes.text)
matplotlib.axes.Axes?
matplotlib.text.Text?
plt.gca?


matplotlib.axes.Axes?
plt.gca().frameon(False)

x = plt.gca().xaxis
l1 = list(dir(plt.gca()))
l2 = list(dir(x))
x?
dir(x)
x.get_ticklabels()






import pandas as pd
import datetime

list(pd.period_range('2015-01-01','2015-12-31',freq= '1M'))







