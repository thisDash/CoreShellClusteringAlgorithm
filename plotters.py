from pathlib import Path
from numpy.core.numeric import NaN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_heatmap(img=np.empty((0,0)), title='', cmap=NaN, show_cbar=True, cbar_labels=[], cbar_ticks=[], min_lat=0, max_lat=0, min_long=0, max_long=0, xlabel='', ylabel=''):
    if len(img) == 0:
        print('First image not provided...')
        return

    if cmap is NaN:
        print('First cmap not provided...')
        return

    if min_lat == max_lat or min_long == max_long:
        print('Something is wrong with the given coordinates...')
        return

    x_axis = np.linspace(0, img.shape[1], 5)
    y_axis = np.linspace(0, img.shape[0], 5)

    if show_cbar:
        if len(cbar_ticks) != 0:
            ax = sns.heatmap(img, cmap=cmap)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks(cbar_ticks)
            colorbar.set_ticklabels(cbar_labels)

        else:
            if len(cbar_labels) == 0:
                min_val = round(np.nanmin(img),2)
                max_val = round(np.nanmax(img),2)
                
                for j1 in np.linspace(min_val, max_val, 10):
                    cbar_labels.append(round(j1,2))

            ax = sns.heatmap(img, cmap=cmap, vmin=cbar_labels[0], vmax=cbar_labels[-1], cbar_kws=dict(ticks=cbar_labels))

    else:
        ax = sns.heatmap(img, cmap=cmap)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([])
        cbar.set_ticklabels([])
    
    cbar = ax.collections[-1].colorbar
    cbar.ax.tick_params(labelsize=20) 
    plt.title(title)
    ax.invert_yaxis()
    ax.set_aspect('equal', 'box')
    plt.yticks(y_axis, np.linspace(min_lat, max_lat, 5), rotation=0, fontsize=22)
    plt.xticks(x_axis, np.linspace(min_long, max_long, 5), rotation=0, fontsize=22)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)

def plot_image(img=np.empty((0,0)), title='', cmap=NaN, show_cbar=True, cbar_labels=None, cbar_ticks=None, min_lat=0, max_lat=0, min_long=0, max_long=0, fig_size=(9,10), path_to_save='', save_name=''):
    if len(img) == 0:
        print('First image not provided...')
        return

    if cmap is NaN:
        print('First cmap not provided...')
        return

    if min_lat == max_lat or min_long == max_long:
        print('Something is wrong with the given coordinates...')
        return

    if cbar_labels is None:
        cbar_labels = []

    if cbar_ticks is None:
        cbar_ticks = []

    fig, axs = plt.subplots(figsize=fig_size)
    x_axis = np.linspace(0, img.shape[1], 5)
    y_axis = np.linspace(0, img.shape[0], 5)

    if show_cbar:
        if len(cbar_ticks) != 0:
            ax = sns.heatmap(img, cmap=cmap)
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks(cbar_ticks)
            colorbar.set_ticklabels(cbar_labels)

        else:
            if len(cbar_labels) == 0:
                min_val = round(np.nanmin(img),2)
                max_val = round(np.nanmax(img),2)

                print(min_val, max_val)
                
                for j1 in np.linspace(min_val, max_val, 10):
                    cbar_labels.append(round(j1,2))

            ax = sns.heatmap(img, cmap=cmap, vmin=cbar_labels[0], vmax=cbar_labels[-1], cbar_kws=dict(ticks=cbar_labels))

    else:
        ax = sns.heatmap(img, cmap=cmap)
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([])
        cbar.set_ticklabels([])

    cbar = ax.collections[-1].colorbar
    cbar.ax.tick_params(labelsize=20) 
    plt.title(title, fontsize=30, pad=20)
    ax.invert_yaxis()
    ax.set_aspect('equal', 'box')
    plt.yticks(y_axis, np.linspace(min_lat, max_lat, 5), rotation=0, fontsize=22)
    plt.xticks(x_axis, np.linspace(min_long, max_long, 5), rotation=0, fontsize=22)
    plt.xlabel("Longitudes", fontsize=24)
    plt.ylabel("Latitudes", fontsize=24)
    fig.tight_layout()
    plt.savefig(Path('{}/{}.jpeg'.format(path_to_save, save_name)))
    plt.close()

#Plots two images, side by side
def plot_two_images(first_img=np.empty((0,0)), first_title='', first_cmap=NaN, show_first_cbar=True, first_cbar_labels=[], first_cbar_ticks=[], second_img=np.empty((0,0)), second_title='', second_cmap=NaN, show_sec_cbar=True, sec_cbar_labels=[], sec_cbar_ticks=[], min_lat=0, max_lat=0, min_long=0, max_long=0, fig_size=(18,10), sup_title='', path_to_save='', save_name=''):

    if len(first_img) == 0:
        print('First image not provided...')
        return

    if len(second_img) == 0:
        print('Second image not provided...')
        return

    if first_cmap is NaN:
        print('First cmap not provided...')
        return

    if second_cmap is NaN:
        print('Second cmap not provided...')
        return

    if min_lat == max_lat or min_long == max_long:
        print('Something is wrong with the given coordinates...')
        return

    print(f'Plotting file {save_name}')

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
    x_axis = np.linspace(0, first_img.shape[1], 5)
    y_axis = np.linspace(0, first_img.shape[0], 5)
    plt.suptitle(sup_title, size=14)

    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        
        if i == 0:
            if show_first_cbar:
                if len(first_cbar_ticks) != 0:
                    ax = sns.heatmap(first_img, cmap=first_cmap)
                    colorbar = ax.collections[0].colorbar
                    colorbar.set_ticks(first_cbar_ticks)
                    colorbar.set_ticklabels(first_cbar_labels)

                else:
                    if len(first_cbar_labels) == 0:
                        min_val = round(np.nanmin(first_img),2)
                        max_val = round(np.nanmax(first_img),2)
                        
                        for j1 in np.linspace(min_val, max_val, 10):
                            first_cbar_labels.append(round(j1,2))

                    ax = sns.heatmap(first_img, cmap=first_cmap, vmin=first_cbar_labels[0], vmax=first_cbar_labels[-1], cbar_kws=dict(ticks=first_cbar_labels))

            else:
                ax = sns.heatmap(first_img, cmap=first_cmap)
                cbar = ax.collections[0].colorbar
                cbar.set_ticks([])
                cbar.set_ticklabels([])
            
            plt.title(first_title, size=14)
            cbar = ax.collections[-1].colorbar
            cbar.ax.tick_params(labelsize=20)

        else:
            if show_sec_cbar:
                if len(sec_cbar_ticks) != 0:
                    ax = sns.heatmap(second_img, cmap=second_cmap)
                    colorbar = ax.collections[0].colorbar
                    colorbar.set_ticks(sec_cbar_ticks)
                    colorbar.set_ticklabels(sec_cbar_labels)
                
                else:
                    if len(sec_cbar_labels) == 0:
                        min_val = round(np.nanmin(second_img),2)
                        max_val = round(np.nanmax(second_img),2)
                        
                        for j2 in np.linspace(min_val, max_val, 10):
                            sec_cbar_labels.append(round(j2,2))

                    ax = sns.heatmap(second_img, cmap=second_cmap, vmin=sec_cbar_labels[0], vmax=sec_cbar_labels[-1], cbar_kws=dict(ticks=sec_cbar_labels))

            else:
                ax = sns.heatmap(second_img, cmap=second_cmap)
                cbar = ax.collections[0].colorbar
                cbar.set_ticks([])
                cbar.set_ticklabels([])
                
            plt.title(second_title, size=14)
            cbar = ax.collections[-1].colorbar
            cbar.ax.tick_params(labelsize=20)
        
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
        
        plt.yticks(y_axis, np.linspace(min_lat, max_lat, 5), rotation=0)
        plt.xticks(x_axis, np.linspace(min_long, max_long, 5), rotation=0)
        plt.xlabel("Longitudes")
        plt.ylabel("Latitudes")

    fig.tight_layout()
    plt.savefig(Path('{}/{}.jpeg'.format(path_to_save, save_name)))
    plt.close()

#Plots two images, side by side
def plot_three_images(first_img=np.empty((0,0)), first_title='', first_cmap=NaN, show_first_cbar=True, first_cbar_labels=[], first_cbar_ticks = [], second_img=np.empty((0,0)), second_title='', second_cmap=NaN, show_sec_cbar=True, sec_cbar_labels=[], sec_cbar_ticks = [], third_img=np.empty((0,0)), third_title='', third_cmap=NaN, show_third_cbar=True, third_cbar_labels=[], third_cbar_ticks = [], min_lat=0, max_lat=0, min_long=0, max_long=0, fig_size=(27,10), path_to_save='', save_name=''):

    if len(first_img) == 0:
        print('First image not provided...')
        return

    if len(second_img) == 0:
        print('Second image not provided...')
        return

    if len(third_img) == 0:
        print('Third image not provided...')
        return

    if first_cmap is NaN:
        print('First cmap not provided...')
        return

    if second_cmap is NaN:
        print('Second cmap not provided...')
        return

    if third_cmap is NaN:
        print('Third cmap not provided...')
        return

    if min_lat == max_lat or min_long == max_long:
        print('Something is wrong with the given coordinates...')
        return

    print(f'Plotting file {save_name}')

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=fig_size)
    x_axis = np.linspace(0, first_img.shape[1], 5)
    y_axis = np.linspace(0, first_img.shape[0], 5)

    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        
        if i == 0:
            if show_first_cbar:
                if len(first_cbar_ticks) != 0:
                    ax = sns.heatmap(first_img, cmap=first_cmap)
                    colorbar = ax.collections[0].colorbar
                    colorbar.set_ticks(first_cbar_ticks)
                    colorbar.set_ticklabels(first_cbar_labels)

                else:
                    if len(first_cbar_labels) == 0:
                        min_val = round(np.nanmin(first_img),2)
                        max_val = round(np.nanmax(first_img),2)
                        
                        for j1 in np.linspace(min_val, max_val, 10):
                            first_cbar_labels.append(round(j1,2))

                    ax = sns.heatmap(first_img, cmap=first_cmap, vmin=first_cbar_labels[0], vmax=first_cbar_labels[-1], cbar_kws=dict(ticks=first_cbar_labels))

            else:
                ax = sns.heatmap(first_img, cmap=first_cmap)
                cbar = ax.collections[0].colorbar
                cbar.set_ticks([])
                cbar.set_ticklabels([])
            
            plt.title(first_title, size=14)
            cbar = ax.collections[-1].colorbar
            cbar.ax.tick_params(labelsize=20)

        elif i == 1:
            if show_sec_cbar:
                if len(sec_cbar_ticks) != 0:
                    ax = sns.heatmap(second_img, cmap=second_cmap)
                    colorbar = ax.collections[0].colorbar
                    colorbar.set_ticks(sec_cbar_ticks)
                    colorbar.set_ticklabels(sec_cbar_labels)

                else:
                    if len(sec_cbar_labels) == 0:
                        min_val = round(np.nanmin(second_img),2)
                        max_val = round(np.nanmax(second_img),2)
                        
                        for j1 in np.linspace(min_val, max_val, 10):
                            sec_cbar_labels.append(round(j1,2))

                    ax = sns.heatmap(second_img, cmap=second_cmap, vmin=sec_cbar_labels[0], vmax=sec_cbar_labels[-1], cbar_kws=dict(ticks=sec_cbar_labels))

            else:
                ax = sns.heatmap(second_img, cmap=second_cmap)
                cbar = ax.collections[0].colorbar
                cbar.set_ticks([])
                cbar.set_ticklabels([])
                
            plt.title(second_title, size=14)
            cbar = ax.collections[-1].colorbar
            cbar.ax.tick_params(labelsize=20)

        else:
            if show_third_cbar:
                if len(third_cbar_ticks) != 0:
                    ax = sns.heatmap(third_img, cmap=third_cmap)
                    colorbar = ax.collections[0].colorbar
                    colorbar.set_ticks(third_cbar_ticks)
                    colorbar.set_ticklabels(third_cbar_labels)

                else:
                    if len(third_cbar_labels) == 0:
                        min_val = round(np.nanmin(third_img),2)
                        max_val = round(np.nanmax(third_img),2)
                        
                        for j1 in np.linspace(min_val, max_val, 10):
                            third_cbar_labels.append(round(j1,2))

                    ax = sns.heatmap(third_img, cmap=third_cmap, vmin=third_cbar_labels[0], vmax=third_cbar_labels[-1], cbar_kws=dict(ticks=third_cbar_labels))

            else:
                ax = sns.heatmap(third_img, cmap=third_cmap)
                cbar = ax.collections[0].colorbar
                cbar.set_ticks([])
                cbar.set_ticklabels([])
                
            plt.title(third_title, size=14)
            cbar = ax.collections[-1].colorbar
            cbar.ax.tick_params(labelsize=20)
        
        ax.invert_yaxis()
        ax.set_aspect('equal', 'box')
        
        plt.yticks(y_axis, np.linspace(min_lat, max_lat, 5), rotation=0, fontsize=22)
        plt.xticks(x_axis, np.linspace(min_long, max_long, 5), rotation=0, fontsize=22)
        plt.xlabel("Longitudes", fontsize=24)
        plt.ylabel("Latitudes", fontsize=24)

    fig.tight_layout()
    plt.savefig(Path('{}/{}.jpeg'.format(path_to_save, save_name)))
    plt.close()