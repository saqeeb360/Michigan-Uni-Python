# Use the following data for this assignment:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,400000,6000), 
                   np.random.normal(43000,400000,6000), 
                   np.random.normal(43500,240000,6000), 
                   np.random.normal(48000,300000,6000)], 
                  index=[1992,1993,1994,1995])
# Cal mean and std
mean_ = []
std_ = []
for i in range(4):
    mean_.append(np.mean(df.iloc[i,:3650]))
    std_.append(np.std(df.iloc[i,:3650]))

# Cal err values
from scipy import stats
z = stats.t.ppf(1-0.025,3649)

err = [z*i/(3650**0.5) for i in std_]
# fig and axes
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)
plt.subplots_adjust(top=0.85)

ax.axis([0,5,0,70000])
# color map : coolwarm
my_cmap = plt.cm.get_cmap('coolwarm')
iy = 40000


# Calculate prob
data_color = []
for i,m in enumerate(mean_):
    temp = stats.norm.cdf(iy,loc = m, scale = err[i]/z )
    data_color.append(temp)
colors = my_cmap(data_color)

# Bar plot
bars = ax.bar([1,2,3,4],mean_,width=0.98, color=colors,alpha = 0.70,
              edgecolor='black', capsize = 10, yerr = err,
              tick_label=['1992','1993','1994','1995']      )
ax.tick_params(bottom='off')




# Horizontal line with text
line1 = ax.axhline(40000, color='orange', linewidth=2)
txt = ax.text(1,65000, 'y=40000', color = 'black')
n_txt = ax.text(3,65000,'n=3650',color='black')
ax.set_title('Sample-oriented task-driven visualizations')

#Check this link to make a linearsegmented coor map
#https://matplotlib.org/examples/pylab_examples/custom_cmap.html
# Colorbar
from matplotlib.cm import ScalarMappable
#my_cmap = plt.cm.get_cmap('coolwarm')
sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0,1))
sm.set_array([])
#cbar = fig.colorbar(sm,orientation= 'horizontal', drawedges=True,values=np.linspace(0,1,10).tolist())
cbar = fig.colorbar(sm,orientation= 'vertical',ticks=np.arange(0,1.1,.1).tolist(), spacing = 'proportional')
cbar.set_label('Probability', rotation=270, labelpad=15)

# OnClick event
def onclick(event):
    #print('you pressed key {0} in ax {1}'.format( event.key, event.inaxes ))
    if event.inaxes not in [ax]:
        return 0
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global line1, txt

    #line1 = ax.axhline(iy, color='orange', linewidth=2)
    txt.set_text('y=%1.0f' % (iy))
    global bars
    for i, bar in enumerate(bars):
        temp = stats.norm.cdf(iy,loc = mean_[i], scale = err[i]/z )
        bars[i].set_color(my_cmap(temp))
    line1.set_ydata(iy)
    fig.canvas.draw_idle()
    
# Connection 
cid = fig.canvas.mpl_connect('button_press_event', onclick)         
from matplotlib.widgets import Slider, Button, RadioButtons
axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([ 0.22,0.94,.47, 0.015], facecolor=axcolor)
sfreq = Slider(axfreq, 'Freq', 1000, 5999, valinit=3650, valfmt='%1.0f')
def update(val):
    global df,ax,n,iy,line1,txt,n_txt
    n = int(sfreq.val) 
    global mean_,std_
    mean_ = list()
    std_ = list()
    ax.cla()
    ax.axis([0,5,0,70000])
    line1 = ax.axhline(iy, color='orange', linewidth=2)
    ax.set_title('Sample-oriented task-driven visualizations')
    txt = ax.text(1,65000, 'y=%d' % (iy), color = 'black')
    n_txt = ax.text(3,65000,'n=%d'%(n),color='black')
    for i in range(0,4):    
        mean_.append(np.mean(df.iloc[i,:n]))
        std_.append(np.std(df.iloc[i,:n]))
    # Cal err values
    global err,z,bars
    err = [z*i/(3650**0.5) for i in std_]
    global data_color
    data_color = []
    for i,m in enumerate(mean_):
        temp = stats.norm.cdf(iy,loc = m, scale = err[i]/z )
        data_color.append(temp)
    colors = my_cmap(data_color)

    bars = ax.bar([1,2,3,4],mean_,width=0.98,alpha = 0.70,color=colors,
              edgecolor='black', capsize = 10, yerr = err,
              tick_label=['1992','1993','1994','1995']      )
    ax.tick_params(bottom='off')        
    
sfreq.on_changed(update)



# add animation with buttom
















