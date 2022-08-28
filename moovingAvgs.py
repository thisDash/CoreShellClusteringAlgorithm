import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from matplotlib.colors import LinearSegmentedColormap
from plotters import plot_image
from processingTools import clean_img
from fileUtils import get_grd_file
from collections import OrderedDict
from scipy.io import savemat
from pathlib import Path

#Generates all "moving averages" of all the possible time intervals 0 <= a <= b <= T
#T being the number of original time steps or generates a single moving average with
#a given window size.
# NOTE: Images file name need to have the number of it at the end in order to get a specific interval only!
def get_averages(path, path_to_save_png='', path_to_save_mat='', min_lat=0, max_lat=0, min_long=0, max_long=0, to_plot=False, window_size=0, min_number=0, max_number=0, fig_size=(9,10)):
    time_series = []
    file_names = []

    dirs = os.listdir(path)
    dirs.sort(key=natural_keys)

    print('###############################')
    print('Getting Averages')
    print('###############################')
    print()

    for filename in dirs:
        file_path = Path('{}/{}'.format(path, filename))
        name, form = os.path.splitext(filename)

        if form == '.mat':
            temps = np.array(scipy.io.loadmat(file_path)['imagem'])

        elif form == '.grd':
            temps, _, _ = get_grd_file(file_path=file_path, file_name=name, path_to_save='')

        else:
            print('Format not supported')
            continue

        if min_number != max_number:
            n = int(name.split('_')[-1])
            print(n)

            if n < min_number or n > max_number:
                continue

        print(f'File: {filename}')

        for l, line in enumerate(temps):
            for c, temp in enumerate(line):
                if isinstance(temp, np.floating):
                    temps[l,c] = temp
                else:
                    temps[l,c] = np.nan
                    
        clean_img(temps)
        time_series.append(temps)
        file_names.append(name)
        print()

    start = 0

    if window_size != 0:
        end = window_size - 1
        window_size = end - start + 1
        print(f'Computing window size of {window_size}')
        return __get_average(time_series, start, end, to_plot, path_to_save_png, path_to_save_mat, min_lat, max_lat, min_long, max_long, fig_size), file_names

    else:
        output = OrderedDict()
        end = 1

        #Will be increasing the window size
        while end < len(time_series):
            window_size = end - start + 1
            print(f'Computing window size of {window_size}')
            output[end + 1] = __get_average(time_series, start, end, to_plot, path_to_save_png, '', min_lat, max_lat, min_long, max_long, fig_size)
            start = 0
            end += 1

        print()

        #To access data: output[time interval size][number of the time series].data
        return output, file_names
    
#TODO Change this
def __get_average(time_series, start, end, to_plot, path_to_save_png, path_to_save_mat, min_lat, max_lat, min_long, max_long, fig_size):
    #cmap = LinearSegmentedColormap.from_list("br", ["darkred", "red", "yellow", "cyan", "blue", "midnightblue"], N=192)
    if path_to_save_mat:
        Path(path_to_save_mat).mkdir(parents=True, exist_ok=True)

    cmap = LinearSegmentedColormap.from_list("br", ["midnightblue", "blue", "cyan", "yellow", "red", "darkred"], N=192)
    averages = []
    moving_end = end
    window_size = end - start + 1

    #Will be doing a moving average with the current window size
    while moving_end < len(time_series):
        print(f'Index: {start}')
        to_mean = []
        to_mean.append(time_series[start])

        #Will sum all matrices in the current window
        for i in np.arange(start+1, moving_end+1):
            to_mean.append(time_series[i])

        res = np.nanmean(np.array(to_mean), axis=0)
        averages.append(res)

        if to_plot:
            folder = Path(f'{path_to_save_png}')

            if not os.path.exists(folder):
                os.makedirs(folder)

            plot_image(img=res,
                       cmap=cmap,
                       min_lat=min_lat,
                       max_lat=max_lat,
                       min_long=min_long,
                       max_long=max_long,
                       fig_size=fig_size,
                       path_to_save=folder,
                       save_name=f'window_{window_size}_n_{start}')
    
            #np.savetxt(f'{folder}/window_{window_size}_n_{start}.csv', res, delimiter=",")

        if path_to_save_mat:
            print('Saving mat file')
            to_save = {"imagem": res}
            savemat(Path(f'{path_to_save_mat}/window_{window_size}_n_{start}.mat'), to_save)      

        start += 1
        moving_end += 1

    return averages

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def plot_statistics(imgs_path, to_save_path, year, window_size):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
    plt.suptitle(f'Year {year} moving averages with W={window_size}')
    avgs = []
    stdevs = []
    medians = []

    dirs = os.listdir(imgs_path)
    dirs.sort(key=natural_keys)
        
    for filename in dirs:
        print(filename)
        file_path = Path('{}/{}'.format(imgs_path, filename))
        _, form = os.path.splitext(filename)

        if form == '.mat':
            temps = np.array(scipy.io.loadmat(file_path)['imagem'])
            avgs.append(np.round(np.nanmean(temps),2))
            stdevs.append(np.round(np.nanstd(temps),2))
            medians.append(np.round(np.nanmedian(temps),2))

        else:
            continue

    x = np.arange(1, len(avgs)+1)

    min_val = np.round(min(np.subtract(avgs,stdevs)),1)
    max_val = np.round(max(np.add(avgs,stdevs)),1)
    plt.sca(axs[0])
    plt.title('Mean temperatures and standard deviations')
    plt.grid()
    plt.xticks(x,x)
    plt.yticks(np.linspace(min_val, max_val, 10))
    plt.errorbar(x, avgs, stdevs, marker='.', capsize=3)
    plt.xlabel("Instants")
    plt.ylabel("Temperature (ºC)")

    min_val = min(medians)
    max_val = max(medians)
    plt.sca(axs[1])
    plt.title('Median temperatures')
    plt.plot(x, medians, marker='.')
    plt.grid()
    plt.xticks(x,x)
    plt.yticks(np.linspace(min_val, max_val, 8))
    plt.xlabel('Instants')
    plt.ylabel('Temperature (ºC)')

    fig.tight_layout()
    plt.savefig(Path(f'{to_save_path}/{year}_windsize_{window_size}_stats.png'))
    plt.close()