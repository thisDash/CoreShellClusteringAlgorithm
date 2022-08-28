from pathlib import Path
import numpy as np
import seaborn as sns
import netCDF4 as nc
import os, sys
import scipy.io
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

def get_grd_file(file_path, file_name='', path_to_save='', to_plot=False, position=-1):
    file = nc.Dataset(file_path)

    longitudes = file['longitude'][:]
    min_long = np.min(longitudes)
    max_long = np.max(longitudes)

    latitudes = file['latitude'][:]
    min_lat = np.min(latitudes)
    max_lat = np.max(latitudes)

    if position == -1:
        temps = file['z'][:]

    else:
        temps = file['z'][:][position]

    for l, line in enumerate(temps):
        for c, temp in enumerate(line):
            if(not isinstance(temp, np.floating)):
                temps[l,c] = np.nan

    #print('get_grd_file(): Latitudes range: {} - {}'.format(min_lat, max_lat))
    #print('get_grd_file(): Longitudes range: {} - {}'.format(min_long, max_long))
    #print('get_grd_file(): Temperatures range: {} - {}'.format(np.min(temps), np.max(temps)))

    if path_to_save or to_plot:
        #print('get_grd_file(): Saving png of the current file')
        full_path_to_save = Path('{}/{}.png'.format(path_to_save, file_name))
        save_temps_img(temps, min_lat, max_lat, min_long, max_long, full_path_to_save, to_plot)

    else:
        pass
        #print('get_grd_file(): Not saving neither plotting the current file')

    return temps, longitudes, latitudes

def get_grd_multiple(dir_path, path_to_save, to_plot=False):
    for filename in os.listdir(dir_path):
        file_path = Path('{}/{}'.format(dir_path, filename))
        name, _ = os.path.splitext(filename)
        _, _, _ = get_grd_file(file_path, name, path_to_save, to_plot)    

def save_temps_img(data, min_lat, max_lat, min_long, max_long, path_to_save, to_plot=False):
    fig, ax = plt.subplots(figsize=(9,10))
    cmap = LinearSegmentedColormap.from_list("br", ["midnightblue", "blue", "cyan", "yellow", "red", "darkred"], N=192)
    min_val = round(np.nanmin(data),2)
    max_val = round(np.nanmax(data),2)

    cbar_labels = []
    for j in np.linspace(min_val, max_val, 10):
        cbar_labels.append(round(j,2))

    ax = sns.heatmap(data, cmap=cmap, vmin=cbar_labels[0], vmax=cbar_labels[-1], cbar_kws=dict(ticks=cbar_labels))
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()
    fig.tight_layout()
    lines = np.linspace(0, data.shape[0], 5)
    cols = np.linspace(0, data.shape[1], 5)
    plt.yticks(lines, np.linspace(min_lat, max_lat, 5))
    plt.xticks(cols, np.linspace(min_long, max_long, 5), rotation=0)
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    
    if to_plot:
        print('Plotting')
        plt.show()

    if path_to_save:
        print('Saving file at: {}'.format(path_to_save))
        plt.savefig(path_to_save)

    else:
        print('Path to save image not specified')

    plt.close()
    
def get_mat_images(img_path, save_path, min_lat, max_lat, min_long, max_long):
    for filename in os.listdir(img_path):
        if filename.endswith(".mat"):
            print(filename)
            name, _ = os.path.splitext(filename)
            file_path = Path("{}/{}".format(img_path, filename))
            img_sci = scipy.io.loadmat(file_path)
            img_np = np.array(img_sci['imagem'])
            cmap = LinearSegmentedColormap.from_list("br", ["midnightblue", "blue", "cyan", "yellow", "red", "darkred"], N=192)

            lines = np.linspace(0, img_np.shape[0], 5)
            cols = np.linspace(0, img_np.shape[1], 5)

            min_val = round(np.nanmin(img_np),2)
            max_val = round(np.nanmax(img_np),2)

            cbar_labels = []
            for j in np.linspace(min_val, max_val, 10):
                cbar_labels.append(round(j,2))

            ax = sns.heatmap(img_np, cmap=cmap, vmin=cbar_labels[0], vmax=cbar_labels[-1], cbar_kws=dict(ticks=cbar_labels))
            ax.invert_yaxis()

            plt.yticks(lines, np.linspace(min_lat, max_lat, 5), rotation=0)
            plt.xticks(cols, np.linspace(min_long, max_long, 5), rotation=0)
            plt.xlabel("Longitudes")
            plt.ylabel("Latitudes")
            plt.savefig(Path('{}/{}.png'.format(save_path, name)))
            plt.close()

#get_grd_multiple('./ImagesPortugal2021/imgs_grd', './ImagesPortugal2021', False)
#get_mat_images('./ImagesMorocco/mat')
#get_dat_images('./CoastLines')