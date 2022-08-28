import os
import scipy.io
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
import re
import xlsxwriter

from matplotlib.colors import LinearSegmentedColormap
from fileUtils import get_grd_file, save_temps_img
from sstsec import SSTSEC
from scipy.io import savemat
from moovingAvgs import get_averages
from pathlib import Path

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Uses the algorithm SSTSEC to segment a path of images of formats .mat or .grd or a single file of the
# same formats
# Returns segmentations results, original images and file names
def segment_sstsec(imgs_path="", original_path='', original_format='', files=[], file_names=[]):

    print()
    print('###############################')
    print('Segmenting')
    print('###############################')
    print()

    output = []
    originals = []
    names = []

    if len(files) != 0:
        for file, file_name in zip(files, file_names):
            print("Segmenting file {}".format(file_name))
            sstsec = SSTSEC('', file_name, file=file)
            output_sstsec, _, _ = sstsec.segment()

            for x,line in enumerate(file):
                for y,temp in enumerate(line):
                    if np.isnan(temp):
                        output_sstsec[x,y] = 2

            if original_path and original_format:
                file_path = Path(f'{original_path}/{file_name}{original_format}')

                if original_format == '.mat':
                    img_sci = scipy.io.loadmat(file_path)
                    img_np = np.array(img_sci['imagem'])

                elif original_format == '.grd':
                    img_np, _, _ = get_grd_file(file_path, file_name)

                originals.append(img_np)

            else:
                originals.append(file)

            output.append(output_sstsec)

        return output, originals, file_names

    else:
        
        dirs = os.listdir(Path(imgs_path))
        dirs.sort(key=natural_keys)

        for filename in dirs:
            name_form = os.path.splitext(filename)
            name = name_form[0]
            format = name_form[1]

            if format not in ['.mat', '.grd']:
                print(f'segment_sstsec(): file format not supported ({format})...')
                continue

            names.append(name)
            print("Segmenting file {}".format(name))
            file_path = Path("{}/{}".format(imgs_path, filename))

            sstsec = SSTSEC(file_path, name, format)
            output_sstsec, _, _ = sstsec.segment()

            if format == '.mat':
                img_sci = scipy.io.loadmat(file_path)
                file = np.array(img_sci['imagem'])

            for x,line in enumerate(file):
                for y,temp in enumerate(line):
                    if np.isnan(temp):
                        output_sstsec[x,y] = 2

            output.append(output_sstsec)

            if original_path:
                file_path = Path(f'{original_path}/{name}{original_format}')
                format = original_format

                if format == '.mat':
                    img_sci = scipy.io.loadmat(file_path)
                    img_np = np.array(img_sci['imagem'])

                elif format == '.grd':
                    img_np, _, _ = get_grd_file(file_path, name)

                originals.append(img_np)
        
        return output, originals, names

# Computes a 4 neighborhood boundary for a given binary matrix
# Returns the computed boundary in a binary matrix format
def get_boundary_4(map=np.empty((0,0))):
    # map[i,j] == 0: non upwelling
    # map[i,j] == 1: upwelling
    # map[i,j] == 2: land
    n_lines, n_cols = map.shape
    boundary = np.zeros((n_lines, n_cols), dtype=int)

    for l, line in enumerate(map):
        for c, val in enumerate(line):
            if val == 0:
                if l > 0 and map[l-1,c] == 1:
                    boundary[l,c] = 1

                elif l < map.shape[0]-1 and map[l+1,c] == 1:
                    boundary[l,c] = 1

                elif c > 0 and map[l,c-1] == 1:
                    boundary[l,c] = 1

                elif c < map.shape[1]-1 and map[l,c+1] == 1:
                    boundary[l,c] = 1

    return boundary

# Computes a 8 neighborhood boundary for a given binary matrix
# Returns the computed boundary in a binary matrix format
def get_boundary_8(map=np.empty((0,0))):
    # map[i,j] == 0: non upwelling
    # map[i,j] == 1: upwelling
    # map[i,j] == 2: land
    n_lines, n_cols = map.shape
    boundary = np.zeros((n_lines, n_cols), dtype=int)

    for l, line in enumerate(map):
        for c, val in enumerate(line):
            if val == 0:
                if l > 0 and map[l-1,c] == 1:
                    boundary[l,c] = 1

                elif l < map.shape[0]-1 and map[l+1,c] == 1:
                    boundary[l,c] = 1

                elif c > 0 and map[l,c-1] == 1:
                    boundary[l,c] = 1

                elif c < map.shape[1]-1 and map[l,c+1] == 1:
                    boundary[l,c] = 1

                elif l > 0 and c > 0 and map[l-1,c-1] == 1:
                    boundary[l,c] = 1

                elif l < map.shape[0]-1 and c < map.shape[1]-1 and map[l+1,c+1] == 1:
                    boundary[l,c] = 1

                elif c > 0 and l < map.shape[0]-1 and map[l+1,c-1] == 1:
                    boundary[l,c] = 1

                elif c < map.shape[1]-1 and l > 0 and map[l-1,c+1] == 1:
                    boundary[l,c] = 1
     
    return boundary

# Computes the frontier of a given binary matrix
# Returns the computed frontier
def get_frontier(region=np.empty((0,0))):
    frontier = np.zeros((region.shape[0], region.shape[1]))
    for l, line in enumerate(region):
        for c, val in enumerate(line):
            if val == 1 and ((c < region.shape[1]-1 and region[l,c+1] != 1) or (c > 0 and region[l,c-1] != 1) or (l > 0 and region[l-1,c] != 1) or (l < region.shape[0]-1 and region[l+1,c] != 1)):
                frontier[l,c] = 1

    return frontier

# Given SSTSEC segementation matrixes and the original images, computes minimum, average and upwelling temperature
# differences for a given time series and plots such results
def get_segm_timeseries_info(segmented=[], originals=[], get_from_cache=False, original_path='', min_lat=36, max_lat=44, min_long=-13, max_long=-8, path_to_save=''):
    if not path_to_save:
        print('No path to save given...')
        return

    print()
    print('###############################')
    print('Getting timeseries info')
    print('###############################')
    print()

    if get_from_cache:
        dirs = os.listdir(Path('./cacheSegmentations'))
        dirs.sort(key=natural_keys)

        for filename in dirs:
            file_path=Path(f'./cacheSegmentations/{filename}')
            img_sci = scipy.io.loadmat(file_path)
            seg = np.array(img_sci['imagem'])
            segmented.append(seg)

            org_file_path=Path(f'{original_path}/{filename}')
            img_sci = scipy.io.loadmat(org_file_path)
            org = np.array(img_sci['imagem'])
            originals.append(org)

    min_temps = []
    avg_temps = []
    stdev_temps = []

    for i, img in enumerate(segmented):
        print(f'Getting info from file number {i}')
        x_axis = np.linspace(0, img.shape[1], 5)
        y_axis = np.linspace(0, img.shape[0], 5)
        original = originals[i]
        temps = []
        min_temp = 50
        intersection = np.zeros((original.shape[0], original.shape[1]))
        set_diff = np.zeros((original.shape[0], original.shape[1]))

        for l,line in enumerate(original):
            for c,temp in enumerate(line):
                if img[l,c] == 1:
                    temps.append(temp)

                    if temp < min_temp:
                        min_temp = temp

                if np.isnan(temp):
                    intersection[l,c] = 2
                    set_diff[l,c] = 3

        min_temps.append(min_temp)
        avg_temps.append(np.nanmean(temps))
        stdev_temps.append(np.nanstd(temps))

        if i > 0 and i < len(segmented):
            prev = segmented[i-1]
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

            for l2, line2 in enumerate(img):
                for c2, temp2 in enumerate(line2):
                    if temp2 == 1 and prev[l2,c2] == 1:
                        intersection[l2,c2] = 1

                    if temp2 == 1 and prev[l2,c2] != 1:
                        set_diff[l2,c2] = 1

                    elif temp2 != 1 and prev[l2,c2] == 1:
                        set_diff[l2,c2] = 2
                        
            for j, ax in enumerate(axs.flatten()):
                plt.sca(ax)

                if j == 0:
                    cmap = LinearSegmentedColormap.from_list("br", ["blue", "red", "white"], N=3)
                    ax = sns.heatmap(intersection, cmap=cmap)
                    plt.title('Intersection')
                    colorbar = ax.collections[0].colorbar
                    colorbar.set_ticks([0.33, 1, 1.66])
                    colorbar.set_ticklabels(['Non Upwelling', 'Upwelling Intersection', 'Land'])

                else:
                    cmap = LinearSegmentedColormap.from_list("br", ["blue", "green", 'red', 'white'], N=4)
                    ax = sns.heatmap(set_diff, cmap=cmap)
                    plt.title('Set Difference')
                    colorbar = ax.collections[0].colorbar
                    colorbar.set_ticks([0.4, 1.1, 1.85, 2.6])
                    colorbar.set_ticklabels(['Non Upwelling', 'Added', 'Removed', 'Land'])

                ax.invert_yaxis()
                ax.set_aspect('equal', 'box')
                
                plt.yticks(y_axis, np.linspace(min_lat, max_lat, 5), rotation=0)
                plt.xticks(x_axis, np.linspace(min_long, max_long, 5), rotation=0)
                plt.xlabel("Longitudes")
                plt.ylabel("Latitudes")

            fig.tight_layout()
            plt.savefig(Path(f'{path_to_save}/upwelling_intersection_setdiff_{i-1}.jpeg'))
            plt.close()

    diff_temps = np.subtract(avg_temps[1:len(avg_temps)], avg_temps[0:len(avg_temps)-1])

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30,8))
    x = np.arange(0, len(min_temps))
    plt.grid()

    diff_temps = []

    for i in range(0, len(avg_temps)-1):
        diff_temps.append(avg_temps[i+1]-avg_temps[i])

    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.plot(x, min_temps)
    ax1.set_title('Minimum upwelling temperatures')
    ax1.set_xlabel('Image Number')
    ax1.set_ylabel('T(ºC)', rotation=0)
    ax1.set_xticks(np.linspace(0, len(min_temps)-1, len(min_temps)))
    ax1.set_yticks(np.linspace(np.min(min_temps), np.max(min_temps), 5))
    
    ax2.errorbar(x, avg_temps, stdev_temps, linestyle='None', marker='.', capsize=3)
    ax2.set_title('Average upwelling temperatures and stdevs')
    ax2.set_xlabel('Image Number')
    ax2.set_ylabel('T(ºC)', rotation=0)
    ax2.set_xticks(np.linspace(0, len(avg_temps)-1, len(avg_temps)))

    x = np.arange(0, len(diff_temps))

    ax3.plot(x, diff_temps)
    ax3.set_title('Upwelling temperature differences')
    ax3.set_xlabel('Difference Number')
    ax3.set_ylabel('T(ºC)', rotation=0)
    ax3.set_xticks(np.linspace(0, len(diff_temps)-1, len(diff_temps)))
    ax3.set_yticks(np.linspace(np.min(diff_temps), np.max(diff_temps), 5))

    plt.grid()

    fig.suptitle('Upwelling Information')
    fig.tight_layout()
    plt.savefig(Path(f'{path_to_save}/upwelling_info.jpeg'))
    plt.close()
    print()

# Given a series of segmentation matrixes, computes a heatmap regarding the upwelling frequency for each image pixel
def get_upwelling_frequency(segmentations=[], get_from_cache=False, min_lat=0, max_lat=0, min_long=0, max_long=0, path_to_save='', create_excel=False):
    
    if min_lat == max_lat or min_long == max_long:
        print('Something is wrong with the given coordinates...')
        return

    if not path_to_save:
        print('Path to save not provided...')
        return

    print('###############################')
    print('Getting Frequencies')
    print('###############################')
    print()

    if get_from_cache:
        dirs = os.listdir(Path('./cacheSegmentations'))
        dirs.sort(key=natural_keys)

        for filename in dirs:
            file_path=Path(f'./cacheSegmentations/{filename}')
            img_sci = scipy.io.loadmat(file_path)
            seg = np.array(img_sci['imagem'])
            segmentations.append(seg)

    elif len(segmentations) == 0:
        print('Segmentations not provided...')
        return

    frequencies = np.zeros((segmentations[0].shape[0], segmentations[0].shape[1]))

    for l, line in enumerate(segmentations[0]):
        for c, n in enumerate(line):
            if n==1:
                frequencies[l,c] += 1

            elif n==2:
                frequencies[l,c] = np.nan

    if len(segmentations) > 1:
        for segmentation in segmentations[1:]:
            for l, line in enumerate(segmentation):
                for c, n in enumerate(line):
                    if n==1:
                        frequencies[l,c] += 1

    fig, _ = plt.subplots(figsize=(6,5))
    cmap = LinearSegmentedColormap.from_list("br", ["midnightblue", "blue", "cyan", "yellow", "red", "darkred"], N=len(segmentations)+1)

    cbar_labels = []

    for j in np.linspace(0, len(segmentations), len(segmentations) + 1):
        #if j % 2 == 0:
       cbar_labels.append(j)

    ax = sns.heatmap(frequencies, cmap=cmap, vmin=cbar_labels[0], vmax=cbar_labels[-1], cbar_kws=dict(ticks=cbar_labels))
    ax.invert_yaxis()
    ax.set_aspect('equal', 'box')

    x_axis = np.linspace(0, segmentations[0].shape[1], 5)
    y_axis = np.linspace(0, segmentations[0].shape[0], 5)
    plt.yticks(y_axis, np.linspace(min_lat, max_lat, 5), rotation=0)
    plt.xticks(x_axis, np.linspace(min_long, max_long, 5), rotation=0)
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.title('Pixel Upwelling Frequency')

    fig.tight_layout()
    plt.savefig(Path(f'{path_to_save}/frequencies_heatmap.png'))
    plt.close()

    if create_excel:
        __create_frequencies_excel(min_lat, max_lat, min_long, max_long, frequencies, path_to_save)

# Creates a frequency .xlsx file
def __create_frequencies_excel(min_lat, max_lat, min_long, max_long, frequencies, path_to_save):
    workbook = xlsxwriter.Workbook(Path(f'{path_to_save}/upwelling_frequencies.xlsx'))
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, 'Longitude')
    worksheet.write(0, 1, 'Latitude')
    worksheet.write(0, 2, 'Frequency')

    row_excel = 1

    lats = np.linspace(min_lat, max_lat, frequencies.shape[0])
    longs = np.linspace(min_long, max_long, frequencies.shape[1])

    for c, col in enumerate(frequencies.T):
        for l, freq in enumerate(col):
            if not np.isnan(freq):
                worksheet.write(row_excel, 0, longs[c])
                worksheet.write(row_excel, 1, lats[l])
                worksheet.write(row_excel, 2, freq)
                row_excel += 1
        
    workbook.close()

# Gets a grd matrix and cuts the file given the latitudes and longitudes
def cut_grd_file(file_path, file_name, min_lat, max_lat, min_long, max_long, position=-1, path_to_save='', to_plot=False):
    temps, longs, lats = get_grd_file(file_path, file_name, position=position)

    min_line = max_line = min_col = max_col = 0

    for l, long in enumerate(longs):
        if long > min_long and min_col == 0:
            min_col = l - 1

        elif long > max_long and max_col == 0:
            max_col = l

    for l2, lat in enumerate(lats):
        if lat > min_lat and min_line == 0:
            min_line = l2 - 1

        elif lat > max_lat and max_line == 0:
            max_line = l2

    return temps[min_line:max_line,min_col:max_col]
    
# Gets a series of files from a .nc file and cuts each file given the latitudes and longitudes
def cut_time_series(path, min_lat, max_lat, min_long, max_long, path_to_save='', path_to_save_mat=''):
    for filename in os.listdir(path):
        if '.nc' in filename:
            print(f'Cutting images of file: {filename}')
            file_path = Path(f'{path}/{filename}')
            file = nc.Dataset(file_path)
            name, _ = os.path.splitext(filename)

            for i in range(file['z'][:].shape[0]):
                temps = cut_grd_file(file_path, name, min_lat, max_lat, min_long, max_long, i)
                full_path_to_save = Path('{}/{}.png'.format(path_to_save, f'{name}_{i}'))
                save_temps_img(temps, min_lat, max_lat, min_long, max_long, full_path_to_save)
                full_path_to_save_mat = Path('{}/{}.mat'.format(path_to_save_mat, f'{name}_{i}'))

                to_save = {"imagem": temps.data}
                savemat(full_path_to_save_mat, to_save)

# Computes quality of images in time series based on Nan values
def analyze_time_series(path):
    path = Path(path)
    n_files = len(os.listdir(path))
    totals = []

    year = [int(s) for s in path.split('/') if s.isdigit()][0]

    for filename in os.listdir(path):
        if '.mat' in filename:
            file_path = Path(f'{path}/{filename}')
            map = np.array(scipy.io.loadmat(file_path)['imagem'])
            total = 0
            for line in map:
                total += np.count_nonzero(np.isnan(line))

            totals.append(total)

    print()
    print(f'Year {year}:\nn_files = {n_files};\navg = {np.sum(totals) / n_files} nans;\nstd_dev = {statistics.stdev(totals)}')
    print()

# Given a path and a window size, a moving average filter is applied to a time series and the resulting
# averaged images are segmented using the SSTSEC algorithm
def segment_time_series(path, window_size, path_to_save, min_lat, max_lat, min_long, max_long, fig_size=(12,5)):
    averages, file_names = get_averages(path, window_size=window_size)

    for i, avg in enumerate(averages):
        name = file_names[i]
        segment_sstsec('', path_to_save, min_lat, max_lat, min_long, max_long, fig_size, file=avg, file_name=name)

# Given a set of original images and segmentations, computes a time series of average upwelling temperatures
def get_temperature_time_series(binary_maps, original_images):
    mean_temps = []

    if len(binary_maps) == 0 or len(original_images) == 0 or len(binary_maps) != len(original_images):
        print('Something is wrong with provided data:')
        print(f'Length of binary maps: {len(binary_maps)}')
        print(f'Length of original images: {len(original_images)}')
        exit()

    for i, binary_map in enumerate(binary_maps):
        original = original_images[i]
        mean_temps.append(np.nanmean(original[binary_map]))

    return mean_temps

# Given a set of SSTSEC segmentations and the pixel resolution of the used images, computes the areas of such upwelling areas
def get_areas_time_series(binary_maps, pixel_res):
    areas = []

    if len(binary_maps) == 0:
        print('Something is wrong with provided data:')
        print(f'Length of binary maps: {len(binary_maps)}')
        exit()

    for binary_map in binary_maps:
        areas.append(np.count_nonzero(binary_map) * pixel_res * pixel_res)

    return areas

# Given a path where a set of .mat files are, these are converted to matrices 
# and returned
def get_files_from_mat(mat_path, type=None):
    files = []
    names = []
    dirs = os.listdir(mat_path)
    dirs.sort(key=natural_keys)

    for filename in dirs:
        name, ext = os.path.splitext(filename)

        if ext == '.mat':
            file_path=Path(f'{mat_path}/{filename}')
            img_sci = scipy.io.loadmat(file_path)
            file = np.array(img_sci['imagem'])
            if type is not None:
                file = file.astype(type)
            files.append(file)
            names.append(name)

    return files, names

def kulczynski_multi(segmentation, ground_truths):
    final_result = 0

    for ground_truth in ground_truths:
        intersection_0 = len(get_intersection(segmentation,ground_truth,0))
        intersection_1 = len(get_intersection(segmentation,ground_truth,1))
        segm_minus_truth_0 = len(get_diff(segmentation,ground_truth,0))
        truth_minus_segm_0 = len(get_diff(ground_truth,segmentation,0))
        segm_minus_truth_1 = len(get_diff(segmentation,ground_truth,1))
        truth_minus_segm_1 = len(get_diff(ground_truth,segmentation,1))

        ks_0 = ((intersection_0/(intersection_0 + segm_minus_truth_0)) + (intersection_0/(intersection_0+truth_minus_segm_0)))/2
        ks_1 = ((intersection_1/(intersection_1 + segm_minus_truth_1)) + (intersection_1/(intersection_1+truth_minus_segm_1)))/2
        final_result += (ks_0+ks_1)/2

    return (final_result / len(ground_truths))

def get_intersection(map_1, map_2, value):
    intersection = []

    for l, line in enumerate(map_1):
        for c, val in enumerate(line):
            if val == value and val == map_2[l,c]:
                intersection.append((l,c))
        
    return intersection

def get_diff(map_1, map_2, value):
    diff = []

    for l, line in enumerate(map_1):
        for c, val in enumerate(line):
            if val == value and val != map_2[l,c]:
                diff.append((l,c))

    return diff

def find_upwelling_front(starting_point, upwelling_frontier, upwelling_area, segm):
    visited = set()
    to_visit = [starting_point]

    while(len(to_visit) > 0):
        to_visit_aux = []
        for (l,c) in to_visit:
            if (l,c) in visited:
                continue
            visited.add((l,c))
            upwelling_area[l,c] = 0
            
            if l > 0:
                if segm[l-1,c] == 1:
                    upwelling_frontier[l-1,c] = 1
                elif (l-1,c) not in visited and (l-1,c) not in to_visit_aux:
                    to_visit_aux.append((l-1,c))

            if l < upwelling_frontier.shape[0]-1:
                if segm[l+1,c] == 1:
                    upwelling_frontier[l+1,c] = 1
                elif (l+1,c) not in visited and (l+1,c) not in to_visit_aux:
                    to_visit_aux.append((l+1,c))

            if c > 0:
                if segm[l,c-1] == 1:
                    upwelling_frontier[l,c-1] = 1
                elif (l,c-1) not in visited and (l,c-1) not in to_visit_aux:
                    to_visit_aux.append((l,c-1))
    
            if c < upwelling_frontier.shape[1]-1:
                if segm[l,c+1] == 1:
                    upwelling_frontier[l,c+1] = 1
                elif (l,c+1) not in visited and (l,c+1) not in to_visit_aux:
                    to_visit_aux.append((l,c+1))

        to_visit = to_visit_aux

# Finds the land pixels region, this being the biggest Nan connected region
def get_land_pixels(img):
    print('##########')
    print()
    print('Finding the land pixels...')
    print()

    land = np.zeros(img.shape, dtype=int)
    visited = np.zeros(img.shape, dtype=int)
    lines_nans, cols_nans = np.where(np.isnan(img))

    for l, c in zip(lines_nans,cols_nans):
        if not visited[l,c]:
            to_visit = [(l,c)]
            land_candidates = np.zeros(img.shape, dtype=int)

            while(len(to_visit) > 0):
                for (l1,c1) in to_visit:
                    to_visit.remove((l1,c1))
                    land_candidates[l1,c1] = 1

                    if l1-1 >= 0 and np.isnan(img[l1-1,c1]) and (l1-1,c1) not in to_visit and not land_candidates[l1-1,c1]:
                        to_visit.append((l1-1,c1))

                    if l1+1 < img.shape[0] and np.isnan(img[l1+1,c1]) and (l1+1,c1) not in to_visit and not land_candidates[l1+1,c1]:
                        to_visit.append((l1+1,c1))

                    if c1-1 >= 0 and np.isnan(img[l1,c1-1]) and (l1,c1-1) not in to_visit and not land_candidates[l1,c1-1]:
                        to_visit.append((l1,c1-1))

                    if c1+1 < img.shape[1] and np.isnan(img[l1,c1+1]) and (l1,c1+1) not in to_visit and not land_candidates[l1,c1+1]:
                        to_visit.append((l1,c1+1))

                if np.count_nonzero(land_candidates) > np.count_nonzero(land):
                    land = land_candidates

                visited = visited | land_candidates

    return land

def kulczynski(u, v):
    a = np.count_nonzero(u) + np.count_nonzero(v)
    aux = u-v
    neg_inds = aux == -1
    pos_inds = aux != -1
    aux[neg_inds] = 1
    aux[pos_inds] = 0
    b = np.count_nonzero(aux)
    aux = v-u
    neg_inds = aux == -1
    pos_inds = aux != -1
    aux[neg_inds] = 1
    aux[pos_inds] = 0
    c = np.count_nonzero(aux)

    return ((a/(a+b))+(a/(a+c)))/2