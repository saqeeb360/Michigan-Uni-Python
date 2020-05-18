# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:29:27 2020

@author: Saqeeb
Week1
"""

"""
# python function

"""
def add_n(x,y,z):
   return x+y+z 
add_n(1,2,3)

def add_n(x,y,z=None):
    if z==None:
        return x+y
    else:
        return x+y+z 
add_n(1,2,3)
add_n(1,2)
a = add_n
def do_math(a, b, kind='add'):
  if (kind=='add'):
    return a+b
  else:
    return a-b

do_math(1, 2,kind='sub')

"""
#python type and sequence
"""

x = (1,'b',3,5.5)
y = [1,'b',4,2.4]
z = (3)  #type int
z = (3,) #type tuple

"""
x = (1, 'b' ,4 ,6.5 ,'a')

x[-3:-5] = ??
x[-4:-3] = ??
x[-3:-5:-1] = ??
x[-3:-1:-1] = ??

x = (1,2,3)
y = (3)

type(x) = ??
type(y) = ??

assign a tuple variable which has only one integer in it.
z = (1,2) #it has 2 integers
"""

for item in x:
    print(item)

(1,2)+(3,4)
(1,2)*3
1 in (1,2,3)

[1,2]+[3,4]
[1,2]*3
1 in [1,2,3]

x[-2:-1]

#substring
x = 'abc xyz ghi'
a = x.split(' ')[0]
 #type(a) is str 
 
 #dict
x = {"a" : 123}

#unpacking
x = ['abc', 'def' , 'ghi']
a ,b ,c = x 
 
 
"""
more on string
"""

sales_record = {'price' : 3.24,
                'num_items' : 4,
                'person' : 'Chris'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'
print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['person'],
                             sales_record['num_items']*sales_record['price']))


"""
Reading and writing CSV file
"""

import csv

%precision 2
with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))
    
mpg
mpg[0].keys()
mpg[0].values()
mpg[0].items()

sum([float(d['cty']) for d in mpg])/len(mpg)
sum([float(d['hwy']) for d in mpg])/len(mpg)

cylinders = set(d['cyl'] for d in mpg)
cylinders

"""
python dates and times
"""
import datetime as dt
import time as tm
tm.time()
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow.year, dtnow.month , dtnow.day


"""
Classes, Objects, map
"""

class Person: #Camel case naming
    department = 'School of Information'
    
    def set_name(self,new_name):
        self.name = new_name
    def set_location(self,new_location):
        self.location = new_location
        
person = Person() 
person.set_name('Saqeeb')
person.set_location('Patna')
print('{} lives in {} and study in {}'.format(person.name,person.location,person.department))

s1 = [1,2,3,4,5]
s2 = [2,5,1,2,4]
s = map(min,s1,s2)
s

[i for i in s]

for i in s:
    print(s)

my_func = lambda a,b,c : a+b
my_func(3,7,5)

my_list = [i for i in range(0,10) if i % 2 == 0 ]

lowercase = 'abcdefghijklmnopqrstuvwxyz'
digits = '0123456789'
correct_answer = [a+b+c+d for a in lowercase for b in lowercase for c in digits for d in digits]
correct_answer[:50] # Display first 50 ids

26*26*10*10 # length of correct_answer

"""
Numpy
"""
import numpy as np
my_list = [1,2,3,4]
x = np.array(my_list)
y = np.array([x,[9,8,7,6]])
y
y.shape

n = np.arange(0,30,2)
n
n = n.reshape(3,5)
n

n = np.linspace(0,10,9)
n

np.ones((3,4))
np.zeros((3,2))
np.eye(3)
np.diag(my_list)
np.array([1,2,3]*3)
np.repeat([1,2,3],3)

p = np.ones((2,3),int)
p
np.vstack([p,2*p])
np.hstack([p,3*p])

x = [1,2,3]
y = [6,7,8]
x = np.array(x)
y = np.array(y)
x+y
x*y
x**2
x.dot(y)

z = np.array([y,2*y])
z
z.T
z.dtype
z = z.astype('f')
z.dtype

a = np.array([8,3,1,5,1,6])
a.sum()
a.max()
a.min()
a.mean()
a.std()
a.argmax() # return the indices of the maximum
a.argmin() # return the indices of the minimum

z = np.arange(36)
z
z = z.reshape(6,6)
z
z[2]
z[2:2]
z[2:4]
z[2:6,4:6]
z[z>30]
z[z>30] = 30
z

z2 = z[:3,:3]
z2
z2[:]=0
z2
z

z3 = z.copy()
z3 = z3[:3,:3]
z3[:] = 5
z3
z

#Iterating over arrays
test = np.random.randint(0,10,(3,4))
test
for i in test:
    print(i)

for i in range(len(test)):
    print(test[i])

for index ,row in enumerate(test):
    print('row',index,'is',row)

test2 =test*2
test2

for i, j in zip(test,test2):
    print(i,'+',j,'=',i+j)

# Enumerate
numbers = ['One', 'Two', 'Three', 'Four', 'Five']
i = 0
for number in numbers:
    print(i, number)
    i += 2

for i, number in enumerate(numbers):
    print(i * 2, number)







