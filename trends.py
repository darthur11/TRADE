# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:26:49 2019

@author: Artur.Dossatayev
"""

from functions import find_resistance, find_support

data2 = data



res = find_resistance(data2['Close'])

for i in range(len(res)):
    data2['R'+str(i)]=res[i]

from matplotlib import pyplot as plt
plt.figure(figsize=(15,15))
plt.plot(data2.iloc[:,3:])
plt.savefig('test2png.png', dpi=600)

data2.drop(['Adj Close', 'Volume'],inplace=True, axis=1)

data2[['Close','R1']].plot()
data2
data2['R'+str(1)]=res[0]
data2['R1']=res[0]
res.plot()

def gentrends(x, window=1/3.0, charts=True):
    """
    Returns a Pandas dataframe with support and resistance lines.
    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    import numpy as np
    import pandas as pd

    x = np.array(x)

    if window < 1:
        window = int(window * len(x))

    max1 = np.where(x == max(x))[0][0]  # find the index of the abs max
    min1 = np.where(x == min(x))[0][0]  # find the index of the abs min

    # First the max
    if max1 + window > len(x):
        max2 = max(x[0:(max1 - window)])
    else:
        max2 = max(x[(max1 + window):])

    # Now the min
    if min1 - window < 0:
        min2 = min(x[(min1 + window):])
    else:
        min2 = min(x[0:(min1 - window)])

    # Now find the indices of the secondary extrema
    max2 = np.where(x == max2)[0][0]  # find the index of the 2nd max
    min2 = np.where(x == min2)[0][0]  # find the index of the 2nd min

    # Create & extend the lines
    maxslope = (x[max1] - x[max2]) / (max1 - max2)  # slope between max points
    minslope = (x[min1] - x[min2]) / (min1 - min2)  # slope between min points
    a_max = x[max1] - (maxslope * max1)  # y-intercept for max trendline
    a_min = x[min1] - (minslope * min1)  # y-intercept for min trendline
    b_max = x[max1] + (maxslope * (len(x) - max1))  # extend to last data pt
    b_min = x[min1] + (minslope * (len(x) - min1))  # extend to last data point
    maxline = np.linspace(a_max, b_max, len(x))  # Y values between max's
    minline = np.linspace(a_min, b_min, len(x))  # Y values between min's

    # OUTPUT
    trends = np.transpose(np.array((x, maxline, minline)))
    trends = pd.DataFrame(trends, index=np.arange(0, len(x)),
                          columns=['Data', 'Max Line', 'Min Line'])

    if charts is True:
        from matplotlib.pyplot import plot, grid, show
        plot(trends)
        grid()
        show()

    return trends, maxslope, minslope

def segtrends(x, segments=2, charts=True):
    """
    Turn minitrends to iterative process more easily adaptable to
    implementation in simple trading systems; allows backtesting functionality.
    :param x: One-dimensional data set
    :param window: How long the trendlines should be. If window < 1, then it
                   will be taken as a percentage of the size of the data
    :param charts: Boolean value saying whether to print chart to screen
    """

    import numpy as np
    y = np.array(x['Close'])
#    for i in range(len(y),1200):
#        y = np.append(y, y[len(y)-1])

    # Implement trendlines
    segments = int(segments)
    maxima = np.ones(segments)
    minima = np.ones(segments)
    segsize = int(len(y)/segments)
    for i in range(1, segments+1):
        ind2 = i*segsize
        ind1 = ind2 - segsize
        maxima[i-1] = max(y[ind1:ind2])
        minima[i-1] = min(y[ind1:ind2])

    # Find the indexes of these maxima in the data
    x_maxima = np.ones(segments)
    x_minima = np.ones(segments)
    for i in range(0, segments):
        x_maxima[i] = np.where(y == maxima[i])[0][0]
        x_minima[i] = np.where(y == minima[i])[0][0]

    if charts:        
        import matplotlib.pyplot as plt
        plt.rc('font', size=6)  
        fig = plt.figure(figsize=(8, 6))
        h = [Size.Fixed(0.5), Size.Fixed(7.)]
        v = [Size.Fixed(0.7), Size.Fixed(5.)]
        divider = Divider(fig, (0.0, 0.0, 0., 0.), h, v, aspect=False)
        ax = Axes(fig, divider.get_position())
        ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        fig.add_axes(ax)
        plt.plot(y, linewidth=1)
        plt.grid(True)

    for i in range(0, segments-1):
        maxslope = (maxima[i+1] - maxima[i]) / (x_maxima[i+1] - x_maxima[i])
        a_max = maxima[i] - (maxslope * x_maxima[i])
        b_max = maxima[i] + (maxslope * (len(y)+300 - x_maxima[i]))
        maxline = np.linspace(a_max, b_max, len(y)+300)

        minslope = (minima[i+1] - minima[i]) / (x_minima[i+1] - x_minima[i])
        a_min = minima[i] - (minslope * x_minima[i])
        b_min = minima[i] + (minslope * (len(y)+300 - x_minima[i]))
        minline = np.linspace(a_min, b_min, len(y)+300)

        if charts:
            plt.plot(maxline, 'g', linewidth=0.15)
            plt.plot(minline, 'r', linewidth=.15)


    if charts:
        #plt.show()
        plt.ylim(min(y[-120:]*.9), max(y[-120:]*1.1))
        plt.rc('xtick', labelsize=6)
        plt.xticks(range(0,1200,4),x.index[range(0,910,4)], rotation=90)
        plt.rc('xtick', labelsize=6)
        plt.xlim(800,1000)
#        plt.xlim(datetime.datetime(2016,1,1),datetime.datetime(2022,1,1))
        plt.savefig('C:/git/TRADE/chart.png', dpi = 750)
        plt.close()

    # OUTPUT
    return x_maxima, maxima, x_minima, minima
a,b,c,d = segtrends(data['Close'], segments = 10, charts = True)

gentrends(data['Close'])

levels = supres(data['Low'], data['High'], n=28, min_touches = 2, stat_likeness_percent = 1.5, bounce_percent = 5)

levels

data['Close'][-200:]
data2 = data
data2.index = [x.date() for x in data2.index]

data2

from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


import matplotlib.pyplot as plt
import datetime
import numpy as np
import yfinance as yf
data = yf.download('^NDX','2016-01-01','2019-08-20')

data.index[100].date()

plt.figure()

plt.plot(data['Close'].index, data['Close'], linewidth=1)

segtrends(data['Close'], segments = 10, charts = True)
gentrends(data['Close'])
y = np.array(data['Close'])
for i in range(len(y),1200):
    y = np.append(y, y[900])






for column in levels.columns:
    data[column] = levels[column]



minitrends(data['Close'], window = 1/5, charts = True)






import pandas as pd
def find_maximums(data,increment):
    start = 0
    end = increment
    maximums = pd.Series([])
    for i in range(int(len(data)/increment)):
        maximums = maximums.append(pd.Series(int(data[start:end].max())))
        start += increment
        end += increment
    maximums = list(maximums)
    maximums.sort()
    return maximums

def find_minimums(data,increment):
    start = 0
    end = increment
    minimums = pd.Series([])
    for i in range(int(len(data)/increment)):
        minimums = minimums.append(pd.Series(int(data[start:end].min())))
        start += increment
        end += increment
    minimums = list(minimums)
    minimums.sort()
    return minimums


    cutoff = 4
    increment = 5
    maximums = find_maximums(data=data['Close'],increment=increment)
    histogram = np.histogram(maximums,bins=(int(len(maximums)/increment*increment)))

    histogram_occurences = pd.DataFrame(histogram[0])
    histogram_occurences.columns = ['occurence']
    histogram_splits = pd.DataFrame(histogram[1])
    histogram_splits.columns = ['bins']
    histogram_bins = []
    for x in histogram_splits.index:
        element = []
        if x < len(histogram_splits.index)-1:
            element.append(int(histogram_splits.iloc[x]))
            element.append(int(histogram_splits.iloc[x+1]))
            histogram_bins.append(element)

    histogram_bins = pd.DataFrame(histogram_bins)
    histogram_bins['occurence'] = histogram_occurences
    histogram_bins.columns = ['start','end','occurence']

    histogram_bins = histogram_bins[histogram_bins['occurence'] >= cutoff]
    histogram_bins.index = range(len(histogram_bins))
    data2 = list(data['Close'])
    data2.sort()
    data2 = pd.Series(data2)
    lst_maxser = []
    for i in histogram_bins.index:
        lst_maxser.append(data2[(data2 > histogram_bins['start'][i]) & (data2 < histogram_bins['end'][i])])

    lst_maxser = pd.Series(lst_maxser)

    lst_resistance=[]

    for i in lst_maxser.index:
        lst_resistance.append(np.linspace(lst_maxser[i].median(),lst_maxser[i].median(),925))
    
    resistance_df = pd.DataFrame(lst_resistance)
    resistance_df.columns = ['resistance']
    resistance_df.dropna(inplace=True)
    resistance_df.index = range(len(resistance_df))
    resistance_ser = pd.Series(resistance_df['resistance'])

from matplotlib import pyplot as plt
plt.figure(figsize=(15,15))
plt.plot(list(data['Close']), linewidth=1)
plt.plot(resistance_df.transpose(),linewidth=0.55)
plt.savefig('test2png.png', dpi=500)


resistance_df.transpose().plot()
data2.plot()