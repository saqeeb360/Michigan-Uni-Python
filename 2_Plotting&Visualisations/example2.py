

import pandas as pd
import numpy as np
from scipy import stats
np.random.seed(12345)

df = pd.DataFrame({1992 : np.random.normal(32000,200000,3650), 
                   1993 : np.random.normal(43000,100000,3650), 
                   1994 : np.random.normal(43500,140000,3650),
                   1995 : np.random.normal(48000,70000,3650)}, 
                  )
import matplotlib.pyplot as plt

mean_ = []
std_ = []
for i in range(4):
    mean_.append(np.mean(df.iloc[:,i]))
    std_.append(np.std(df.iloc[:,i]))
z = stats.t.ppf(1-0.025,3649)

err = [z*i/(3650**0.5) for i in std_]

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)


my_cmap = plt.cm.get_cmap('coolwarm')
iy = 40000
data_color = []
for i,m in enumerate(mean_):
    temp = stats.norm.cdf(iy,loc = m, scale = err[i]/z )
    data_color.append(temp)
colors = my_cmap(data_color)

bars = ax.bar([1,2,3,4],mean_,width=0.98, color=colors,alpha=.8,
              edgecolor='black', capsize = 10, yerr = err,
              tick_label=['1992','1993','1994','1995']      )

line1 = ax.axhline(40000, color='orange', linewidth=2)


from matplotlib.cm import ScalarMappable

#my_cmap = plt.cm.get_cmap('coolwarm')
sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,1))
sm.set_array([])
#cbar = fig.colorbar(sm,orientation= 'horizontal', drawedges=True,values=np.linspace(0,1,10).tolist())
cbar = fig.colorbar(sm,orientation= 'vertical',ticks=np.arange(0,1.1,.1).tolist(), spacing = 'proportional')
cbar.set_label('Probability', rotation=270, labelpad=15)

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    #print ('x = %d, y = %d'%(ix, iy))
    #global coords
    #coords.append((ix, iy))
    #if len(coords) == 5:
    #   fig.canvas.mpl_disconnect(cid)
    # check for a line
    global line1
    # remove a line
    line1.remove()
    # plot a line
    line1 = ax.axhline(iy, color='orange', linewidth=2)
    # assign color
    global bars
    for i, bar in enumerate(bars):
        temp = stats.norm.cdf(iy,loc = mean_[i], scale = err[i]/z )
        bars[i].set_color(my_cmap(temp))
    
    
    
cid = fig.canvas.mpl_connect('button_press_event', onclick)         
plt.show()