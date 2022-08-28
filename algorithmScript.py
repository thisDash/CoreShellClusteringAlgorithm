import os
import shutil
import h5py
import numpy as np
import time
import itertools
import progressbar
from scipy.io import savemat
from experimentUtils import *
from plotters import *
from sklearn.metrics.cluster import pair_confusion_matrix
from pathlib import Path
from processingTools import clean_img
from temperatureProcessingV3 import TemperaturePreprocessingV3
from upwellingClumpsBuilder import UpwellingClumpsBuilder

# Change these params #########################################################
pixel_res = 2 # Pixel resolution in km
n_folder = 1 # Folder number to be used in the experiments saving path
year = 2007 # Year of the data to be analysed
min_partition_size = 3 # Minimum partition size of an upwelling clump
to_julia = False # To perform first stage of the preprocessing pipeline
to_average = False # To perform second stage of the preprocessing pipeline
to_preprocess = False # To perform third stage of the preprocessing pipeline
to_segment = True # To segment sst instants using the sstsec algorithm
to_get_from_cache = False # To get sstsec segmentations from cache.
to_plot_segmentation_info = False # Computed set differences of obtaines segmentations
to_plot_comparisons = True # To plot comparisons present in folder comparisons
to_exec_main_alg = True # To exec core-shell clustering algorithm
upwelling_clumps_starts = [0,8,13]
experiment_parent_path = 'Portugal' # Root path for the experiments results
original_files_start = 'sst_week' # Starting names of the original sst grids
imgs_parent_path = 'PortugalImagesSample' # Root path for the original sst grids

# Number of starting original SST grids
n_files = 46

# Do not change from here #####################################################
degrees = u'\N{DEGREE SIGN}'
min_lat = 36 # Minimum latitude of the explored Portuguese region
max_lat = 44 # Maximum latitude of the explored Portuguese region
min_long = -13 # Minimum longitude of the explored Portuguese region
max_long = -8 # Maximum longitude of the explored Portuguese region
figsize_sst = (9,10) # Figsize to plot a single sst grid

# Folder checking
previous_experiment_path = ''

for filename in os.listdir(Path(f'./experimentsSample/{experiment_parent_path}')):
    if filename.startswith(f'{n_folder}_Full'):
        if f'_{year}' in filename:
            print('Found the following folder with the same year and folder number:')
            print(filename)

        else:
            print('Found the following folder with the same folder number:')
            print(filename)

        print('Do you want to delete it? Type Y (yes) or N (no)...')

        while True:
            a = input()
            if a == 'Y':
                print(f'Deleting folder {filename} and creating a new one...')
                shutil.rmtree(Path(f'./experimentsSample/{experiment_parent_path}/{filename}'))
                break
            elif a == 'N':
                print('Not deleting the folder!')
                previous_experiment_path = Path(f'./experimentsSample/{experiment_parent_path}/{filename}')
                break
            else:
                print('Command not recognized, type again...')

# MAT PATHS 
# Do not change these paths ###################################################

if previous_experiment_path:
    if to_exec_main_alg:
        experiment_path = Path(f'./experimentsSample/{experiment_parent_path}/{n_folder}_Full_Experiment_{year}')
        os.rename(previous_experiment_path,experiment_path)

    else:
        experiment_path = previous_experiment_path

else:
    experiment_path = Path(f'./experimentsSample/{experiment_parent_path}/{n_folder}_Full_Experiment_{year}')
    Path(experiment_path).mkdir(parents=True, exist_ok=True)

# Only change these if there is the need to change any authomatic folder naming

julia_images_path = Path(f'{experiment_path}/preprocessing_phase1')
averages_path = Path(f'{experiment_path}/preprocessing_phase2')
averages_originals = Path(f'{experiment_path}/averages_original')
pre_processed_path = Path(f'{experiment_path}/sst_instants')
segmentations_info_path = Path(f'{experiment_path}/sstsec_info')
original_preprocessed_segmented_comp_path = Path(f'{experiment_path}/comparisons')
core_shell_clusters_path = Path(f'{experiment_path}/core_shell_algorithm')
core_shell_plots_path = Path(f'{experiment_path}/core_shell_plots')

julia_mat_path = Path(f'./{imgs_parent_path}/{year}/mat/preprocessing_phase_1')
originals_mat_path = Path(f'./{imgs_parent_path}/{year}/mat')
cache_julia_path = Path(f'./cacheJulia')
averages_mat_path = Path(f'./{imgs_parent_path}/{year}/mat/preprocessing_phase_2')
sst_instants_no_preprocess_mat_path = Path(f'./{imgs_parent_path}/{year}/mat/sst_instants_no_preprocess')
instants_mat_path = Path(f'./{imgs_parent_path}/{year}/mat/sst_instants')
segmented_path = Path(f'{instants_mat_path}/segmentations')

# HELPER CLASSES AND FUNCTIONS

img_cmap = LinearSegmentedColormap.from_list("br", ["midnightblue", "blue", "cyan", "yellow", "red", "darkred"], N=192)
segmentation_cmap = LinearSegmentedColormap.from_list("br", ["dodgerblue", "green", "white"], N=3)

class Decision:

    def __init__(self, r, s, line, col, delta):
        self.r = r
        self.s = s
        self.line = line
        self.col = col
        self.delta = delta

def compute_G(r_count, s_counts, r_intensity, s_intensity):
    output = 0
    r_parcels = np.square(r_intensity) * r_count
    s_parcels = np.multiply(np.square(s_intensity), s_counts)

    for i, r_parcel in enumerate(r_parcels):
        output += r_parcel + s_parcels[i]

    return output

# Given a previous mean and changes, computes new mean of the core
def compute_new_mean_r(change, prev_means, originals, old_count):
    new_means = []
    if change == 1:
        for i, (prev_mean, new_value) in enumerate(zip(prev_means, originals)):
            if np.isnan(new_value):
                new_means.append(prev_means[i])

            else:
                new_means.append(prev_mean + (new_value - prev_mean) / (old_count+1))

    else:
        for i, (prev_mean, rem_value) in enumerate(zip(prev_means, originals)):
            if np.isnan(rem_value):
                new_means.append(prev_means[i])

            else:
                new_means.append((old_count * prev_mean - rem_value) / (old_count-1))

    return new_means

# Given a previous mean and changes, computes new mean of the shells
def compute_new_mean_s(changes, prev_means, originals, old_counts):
    new_means = []

    for i, change in enumerate(changes):
        prev_mean = prev_means[i]
        value = originals[i]
        old_count = old_counts[i]

        if np.isnan(value) or change == 0:
            new_means.append(prev_mean)
        elif change == 1:
            new_means.append(prev_mean + (value - prev_mean) / (old_count+1))
        elif change == -1:
            new_means.append((old_count * prev_mean - value) / (old_count-1))
            
    return new_means

# Computes kulczynski score
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

# JULIA PRE PROCESSING

if to_julia:
    shutil.rmtree(julia_mat_path, ignore_errors=True)
    shutil.rmtree(cache_julia_path, ignore_errors=True)
    shutil.rmtree(julia_images_path, ignore_errors=True)
    os.mkdir(cache_julia_path)
    os.mkdir(julia_mat_path)
    os.mkdir(julia_images_path)

    julia_call = 'using GMT;using MAT;'

    for i in range(0, n_files):
        julia_call += f'a = matopen("{Path(f"./{imgs_parent_path}/{year}/mat/{original_files_start}_{year}_2km_{i}.mat")}"); temps = read(a, "imagem"); trend = GMT.grdtrend(temps, N=3, D=true); file = matopen("{Path(f"./cacheJulia/{original_files_start}_{year}_2km_{i}.mat")}", "w"); write(file, "imagem", trend.z); close(file);'

    julia_call += 'return;'
    os.system(f"julia -e '{julia_call}'")
    
    # JULIA OUTPUT AUXILIARY PROCESSING

    for filename in os.listdir(cache_julia_path):
        name_form = os.path.splitext(filename)
        name = name_form[0]
        format = name_form[1]

        if format != ".mat":
            continue

        print(filename)
        print()
        filepath = Path(f'{cache_julia_path}/{filename}')
        f = h5py.File(filepath)

        for k, v in f.items():
            if k == 'imagem':
                grad = np.array(v)

        grad = grad.T
        clean_img(grad)

        img = {"imagem": grad}
        savemat(Path(f'{julia_mat_path}/{filename}'), img)

        min_val = round(np.nanmin(grad),2)
        max_val = round(np.nanmax(grad),2)
        cbar_labels = []
        for j in np.linspace(min_val, max_val, 10):
            cbar_labels.append(round(j,2))

        plot_image(img=grad,
                cmap=img_cmap,
                fig_size=figsize_sst,
                cbar_labels=cbar_labels,
                min_lat=min_lat,
                max_lat=max_lat,
                min_long=min_long,
                max_long=max_long,
                path_to_save=julia_images_path,
                save_name=filename)


# MOVING AVERAGES COMPUTATION

if to_average:
    shutil.rmtree(averages_path, ignore_errors=True)
    shutil.rmtree(averages_mat_path, ignore_errors=True)
    shutil.rmtree(averages_originals, ignore_errors=True)
    shutil.rmtree(sst_instants_no_preprocess_mat_path, ignore_errors=True)
    Path(averages_path).mkdir(parents=True, exist_ok=True)
    Path(averages_mat_path).mkdir(parents=True, exist_ok=True)
    Path(averages_originals).mkdir(parents=True, exist_ok=True)
    Path(sst_instants_no_preprocess_mat_path).mkdir(parents=True, exist_ok=True)
    
    get_averages(path=julia_mat_path, 
                path_to_save_png=averages_path,
                path_to_save_mat=averages_mat_path, 
                min_lat=min_lat, 
                max_lat=max_lat, 
                min_long=min_long, 
                max_long=max_long, 
                to_plot=True,
                window_size=5,
                min_number=11,
                max_number=37,
                fig_size=figsize_sst)

    get_averages(path=originals_mat_path,
                 path_to_save_png=averages_originals,
                 path_to_save_mat=sst_instants_no_preprocess_mat_path,  
                 min_lat=min_lat, 
                 max_lat=max_lat,
                 min_long=min_long,
                 max_long=max_long,
                 to_plot=True,
                 window_size=5,
                 min_number=11,
                 max_number=37,
                 fig_size=figsize_sst)

# PRE PROCESSING COMPUTATION

if to_preprocess:
    shutil.rmtree(pre_processed_path, ignore_errors=True)
    shutil.rmtree(instants_mat_path, ignore_errors=True)
    Path(pre_processed_path).mkdir(parents=True, exist_ok=True)
    Path(instants_mat_path).mkdir(parents=True, exist_ok=True)
    preprocesser = TemperaturePreprocessingV3(averages_mat_path,
                                                min_lat=min_lat,
                                                max_lat=max_lat,
                                                min_long=min_long,
                                                max_long=max_long)

    preprocesser.preprocess(path_to_save_img=pre_processed_path, 
                            path_to_save_mat=instants_mat_path,
                            path_to_save_land=Path(f'{instants_mat_path}/land'),
                            fig_size=figsize_sst)

# PRE PROCESSED FILES GETTER

pre_processed = []
dirs = os.listdir(instants_mat_path)
dirs.sort(key=natural_keys)

for filename in dirs:
    name, ext = os.path.splitext(filename)

    if ext == '.mat':
        print(name)
        file_path = instants_mat_path / filename
        img_sci = scipy.io.loadmat(file_path)
        aux = np.array(img_sci['imagem'])
        pre_processed.append(aux)

# SSTSEC SEGMENTATION

if to_segment:
    outputs_sstsec, originals, names = segment_sstsec(imgs_path=instants_mat_path, 
                                                      original_path=sst_instants_no_preprocess_mat_path, 
                                                      original_format='.mat')
    Path(segmented_path).mkdir(parents=True, exist_ok=True)

    aux_outputs = []

    lands = []

    dirs = os.listdir(instants_mat_path)
    dirs.sort(key=natural_keys)

    for filename in dirs:
        name, ext = os.path.splitext(filename)

        if ext == '.mat':
            file_path = Path(f'{instants_mat_path}/land/{filename}')
            img_sci = scipy.io.loadmat(file_path)
            aux = np.array(img_sci['imagem'])
            lands.append(aux)

    for i, (instant, segmentation) in enumerate(zip(pre_processed, outputs_sstsec)):
        starting_point = (0,0)
        upwelling_frontier = np.zeros(instant.shape)
        upwelling_area = np.ones(instant.shape)

        find_upwelling_front(starting_point, upwelling_frontier, upwelling_area, segmentation)

        land_lines, land_cols = np.where(lands[i] == 1)
        starting_point = (land_lines[0],land_cols[0])
        find_upwelling_front(starting_point, upwelling_frontier, upwelling_area, segmentation)

        upwelling_area[segmentation == 2] = 2
        aux_outputs.append(upwelling_area)

        name = names[i]
        to_save = {'imagem': upwelling_area}
        savemat(Path(f'{segmented_path}/{name}.mat'), to_save)

    outputs_sstsec = aux_outputs

else:
    if to_get_from_cache:
        path = Path('./cacheSegmentations')

    else:
        path = segmented_path

    dirs = os.listdir(path)
    dirs.sort(key=natural_keys)

    outputs_sstsec = []
    originals = []
    names = []

    for filename in dirs:
        file_path = Path(f'{path}/{filename}')
        name, ext = os.path.splitext(filename)

        if ext == '.mat':
            img_sci = scipy.io.loadmat(file_path)
            aux = np.array(img_sci['imagem'])
            aux = aux.astype(int)
            outputs_sstsec.append(aux)
            names.append(name)

    dirs = os.listdir(sst_instants_no_preprocess_mat_path)
    dirs.sort(key=natural_keys)

    for filename in dirs:
        _, ext = os.path.splitext(filename)

        if ext == '.mat':
            file_path = Path(f'{sst_instants_no_preprocess_mat_path}/{filename}')
            img_sci = scipy.io.loadmat(file_path)
            originals.append(np.array(img_sci['imagem']))

if to_plot_segmentation_info:
    shutil.rmtree(segmentations_info_path, ignore_errors=True)
    Path(segmentations_info_path).mkdir(parents=True, exist_ok=True)
    get_segm_timeseries_info(segmented=outputs_sstsec, 
                            originals=originals, 
                            path_to_save=segmentations_info_path)

# PLOTTING OF ORIGINAL IMAGES, PRE PROCESSED IMAGES AND SEGMENTATION RESULTS

if to_plot_comparisons:
    shutil.rmtree(original_preprocessed_segmented_comp_path, ignore_errors=True)
    Path(original_preprocessed_segmented_comp_path).mkdir(parents=True, exist_ok=True)

    for i, output in enumerate(outputs_sstsec):
        original = originals[i]
        pre_proc = pre_processed[i]
        name = names[i]

        min_val = round(np.nanmin(original),2)
        max_val = round(np.nanmax(original),2)
        original_cbar_labels = []
        for j in np.linspace(min_val, max_val, 10):
            original_cbar_labels.append(round(j,2))

        min_val = round(np.nanmin(pre_proc),2)
        max_val = round(np.nanmax(pre_proc),2)
        proc_cbar_labels = []
        for j in np.linspace(min_val, max_val, 10):
            proc_cbar_labels.append(round(j,2))

        plot_three_images(first_img=original, 
                        first_title='Moving Average without preprocessing', 
                        first_cmap=img_cmap, 
                        show_first_cbar=True, 
                        first_cbar_labels=original_cbar_labels,
                        second_img=pre_proc,
                        second_title='Preprocessed moving average',
                        second_cmap=img_cmap,
                        show_sec_cbar=True,
                        sec_cbar_labels=proc_cbar_labels,
                        third_img=output, 
                        third_title='S-STSEC segmentation', 
                        third_cmap=segmentation_cmap, 
                        show_third_cbar=True,
                        third_cbar_labels=['Non Upwelling', 'Upwelling', 'Nan'],
                        third_cbar_ticks=[0.33, 1, 1.66],
                        min_lat=min_lat, 
                        max_lat=max_lat, 
                        min_long=min_long, 
                        max_long=max_long,
                        fig_size=(figsize_sst[0]*3,figsize_sst[1]),
                        path_to_save=original_preprocessed_segmented_comp_path,
                        save_name=name)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# CORE-SHELL CLUSTERING ------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

if to_exec_main_alg:
    # OBTAINING UPWELLING CLUMPS
    if len(upwelling_clumps_starts) == 0:
        print(f'Obtaining upwelling clumps for year {year}')
        print()
        clumpBuilder = UpwellingClumpsBuilder(segmentations_path=Path(f'./{imgs_parent_path}/{year}/mat/sst_instants/segmentations'),
                                            original_averages_path=Path(f'./{imgs_parent_path}/{year}/mat/sst_instants_no_preprocess'))

        clumps = clumpBuilder.get_clumps(min_lat=min_lat, max_lat=max_lat)

        clumps.sort()
        got_error = False

        for clump in clumps:
            if len(clump) < min_partition_size:
                got_error = True
                break

            else:
                prev_inst = clump[0]
                for inst in clump[1:]:
                        if inst != prev_inst + 1:
                            got_error = True
                            break

                        prev_inst = inst
                
        if got_error:
            print('It was not possible to obtain a correct result from the IAP algorithm')
            print()
            print('IAP Results:')
            for i, clump in enumerate(clumps):
                print(f'Upwelling Clump {i+1}: {clump}')
            print()

            while True:
                error = False
                print("Insert the upwelling clumps starts, separated by a single ',' with an Enter at the end")
                clumps_string = input().strip()
                print()

                if clumps_string and ',' in clumps_string:
                        clumps_starts = []
                        clumps_array = clumps_string.split(',')

                        try:
                            if int(clumps_array[0].strip()) != 0:
                                print('The first upwelling clump should start at instant 0')
                                print()
                                continue
                        except:
                            print('The first upwelling clump should start at instant 0')
                            print()
                            continue

                        try:
                            if int(clumps_array[-1].strip()) > 22:
                                print('The last number is too high, please try again')
                                print()
                                continue
                        except:
                            print('The last number is too high, please try again')
                            print()
                            continue

                        for i, clump_string in enumerate(clumps_array):
                            try:
                                clump_start = int(clump_string.strip())

                                if i > 0 and clump_start < clumps_starts[-1] + 3 or (i == len(clumps_array) - 1 and len(np.arange(clump_start, 23)) < 3):
                                    print('The provided partitions did not respect the minimum partition size')
                                    print()
                                    error = True
                                    break

                                clumps_starts.append(clump_start)
                            except:
                                error = True
                                break
                        
                        if error:
                            continue

                        else:
                            break
            clumps = []
            print('Will work with the following clumps:')
            for i, clump_start in enumerate(clumps_starts):
                if i < len(clumps_starts) - 1:
                    clump = list(np.arange(clump_start, clumps_starts[i+1]))
                        
                else:
                    clump = list(np.arange(clump_start, 23))

                clumps.append(clump)
                print(f'Upwelling Clump {i+1}: {clump}')

            print()

        else:
            print('Obtained Upwelling Clumps:')
            for i, clump in enumerate(clumps):
                print(f'Upwelling Clump {i+1}: {clump}')
            print()

        upwelling_clumps_starts = [clump[0] for clump in clumps]
        print(upwelling_clumps_starts)

    else:
        print('Will work with upwelling clumps provided')

    # PATHS CREATION
    old_experiment_path = experiment_path

    for i in upwelling_clumps_starts:
        experiment_path = Path(f'{experiment_path}_{i+1}')

    os.rename(old_experiment_path,experiment_path)
    original_preprocessed_segmented_comp_path = Path(f'{experiment_path}/comparisons')
    core_shell_clusters_path = Path(f'{experiment_path}/core_shell_algorithm')

    shutil.rmtree(core_shell_clusters_path, ignore_errors=True)
    Path(core_shell_clusters_path).mkdir(parents=True, exist_ok=True)

    # STRUCTURES INITIALIZATIONS
    aris = []
    kulczynski_multi_values = []
    simple_kulczynski_values = []
    cores_intensities = []
    shells_intensities = []
    times = []
    clumps_run_times = []
    g_values = []

    for upwelling_clump, starting_map in enumerate(upwelling_clumps_starts):
        start_time = time.time()

        if upwelling_clump == len(upwelling_clumps_starts)-1:
            ending_map = len(outputs_sstsec)-1

        else:
            ending_map = upwelling_clumps_starts[upwelling_clump+1]-1

        print()
        print(f'Algorithm for Upwelling Clump between {starting_map+1} and {ending_map+1}')
        print()

        n_maps = ending_map - starting_map + 1

        # GETTING SSTSEC OUTPUTS OF SUCH UPWELLING CLUMP
        curr_outputs_sstsec = outputs_sstsec[starting_map:ending_map+1]
        curr_pre_processed = pre_processed[starting_map:ending_map+1]

        n_lines, n_cols = curr_outputs_sstsec[0].shape

        # map with intersection of nan pixels, which will result in an aproximation of the land pixels
        land_int = np.ones((n_lines, n_cols), dtype=int)

        # core R initializations
        # r[i,j] == 0: does not belong to r
        # r[i,j] == 1: belongs to r

        r = np.ones((n_lines, n_cols), dtype=int)

        prev_land = None
        sstsec_union = np.zeros((n_lines, n_cols), dtype=int)
        non_upwelling_union = np.zeros((n_lines, n_cols), dtype=int)
        land_union = np.zeros((n_lines, n_cols), dtype=int)

        for out in curr_outputs_sstsec:
            aux = np.zeros((n_lines, n_cols), dtype=int)
            aux[out == 2] = 1
            land_int = land_int & aux
            land_union = land_union | aux

            non_upwelling = np.zeros((n_lines, n_cols), dtype=int)
            non_upwelling[out==0] = 1
            non_upwelling_union = non_upwelling_union | non_upwelling

            upwelling = np.zeros((n_lines, n_cols), dtype=int)
            upwelling[out == 1] = 1
            sstsec_union = sstsec_union | upwelling

            r = r & upwelling

        upwelling_ls, upwelling_cs = np.where(sstsec_union == 1)

        for l,c in zip(upwelling_ls, upwelling_cs):
            if land_union[l,c] == 1 and non_upwelling_union[l,c] == 0:
                r[l,c] = 1

        # shells S initializations
        s = []

        # if out[i,j] == 0 and r[i,j] == 0 || r[i,j] == 1 -> not in s
        # if out[i,j] == 1 and r[i,j] == 1, out[i,j] - r[i,j] == 0 -> not in s
        # if out[i,j] == 1 and r[i,j] == 0, out[i,j] - r[i,j] == 1 -> in s
        for out in curr_outputs_sstsec:
            aux_out = np.zeros((n_lines, n_cols), dtype=int)
            upwelling_inds = out == 1
            aux_out[upwelling_inds] = 1
            shell = aux_out - r

            ind_not_1 = shell != 1
            shell[ind_not_1] = 0

            # Cleaning process of pixels near the coast

            for l, line in enumerate(shell):
                for c, v in enumerate(line):
                    if v == 1:
                        try:
                            if land_int[l, c+1] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

                        try:
                            if land_int[l, c-1] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

                        try:
                            if land_int[l+1, c] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

                        try:
                            if land_int[l-1, c] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

                        try:
                            if land_int[l+1, c-1] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

                        try:
                            if land_int[l+1, c+1] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

                        try:
                            if land_int[l-1, c-1] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

                        try:
                            if land_int[l-1, c+1] == 1:
                                shell[l,c] = 0
                        except Exception:
                            continue

            s.append(shell)

        # set B initializations
        b = np.copy(r)

        for shell in s:
            b = b | shell

        land_inds = land_int == 1
        b_and_land = np.copy(b)
        b_and_land[land_inds] = 2
        b = b | get_boundary_4(b_and_land)

        # Algorithm loop
        decisions = []

        r_bool = r.astype(bool)
        s_bool = [shell.astype(bool) for shell in s]

        combinations = [list(i) for i in itertools.product([0, 1], repeat=n_maps)]

        r_old_intensities = [np.nanmean(pre_proc[r_bool]) for pre_proc in curr_pre_processed]
        s_old_intensities = [np.nanmean(pre_proc[s_bool[i]]) for i, pre_proc in enumerate(curr_pre_processed)]
        r_old_count = np.count_nonzero(r_bool)
        s_old_count = [np.count_nonzero(shell) for shell in s_bool]

        g_old = compute_G(r_old_count, s_old_count, r_old_intensities, s_old_intensities)

        n_b = np.count_nonzero(b)
        current = 0
        bar = progressbar.ProgressBar(maxval=n_b, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for l, line in enumerate(b):
            for c, val_b in enumerate(line):
                if val_b == 1:
                    values = [pre_proc[l,c] for pre_proc in curr_pre_processed]

                    # case of r[i,j] being 1
                    r_new_count = r_old_count

                    if r[l,c] == 0:
                        r_intensity = compute_new_mean_r(1, r_old_intensities, values, r_old_count)
                        r_new_count += 1

                    else:
                        r_intensity = r_old_intensities
                        r_new_count = r_old_count

                    changes_s = []
                    s_new_counts = []

                    for i, shell in enumerate(s):
                        if s[i][l,c] == 1:
                            changes_s.append(-1)
                            s_new_counts.append(s_old_count[i]-1)

                        else:
                            changes_s.append(0)
                            s_new_counts.append(s_old_count[i])

                    s_intensity = compute_new_mean_s(changes_s, s_old_intensities, values, s_old_count)
                    g_new = compute_G(r_new_count, s_new_counts, r_intensity, s_intensity)
                    highest_delta = g_new - g_old

                    d = Decision(r=1, s=np.zeros(len(s)), line=l, col=c, delta=highest_delta)

                    # case of r[i,j] being 0
                    r_new_count = r_old_count

                    if r[l,c] == 1:
                        r_intensity = compute_new_mean_r(-1, r_old_intensities, values, r_old_count)
                        r_new_count -= 1

                    else:
                        r_intensity = r_old_intensities

                    for comb in combinations:
                        s_new_counts = []
                        changes_s = []

                        for i, value in enumerate(comb):
                            if s[i][l,c] == 1 and value == 0:
                                changes_s.append(-1)
                                s_new_counts.append(s_old_count[i]-1)

                            elif s[i][l,c] == 0 and value == 1:
                                changes_s.append(1)
                                s_new_counts.append(s_old_count[i]+1)

                            else:
                                changes_s.append(0)
                                s_new_counts.append(s_old_count[i])
                                
                        s_intensity = compute_new_mean_s(changes_s, s_old_intensities, values, s_old_count)
                        g_new = compute_G(r_new_count, s_new_counts, r_intensity, s_intensity)
                        delta = g_new - g_old

                        if delta > highest_delta:
                            d = Decision(r=0, s=comb, line=l, col=c, delta=delta)
                            highest_delta = delta

                    # checking if greatest delta can be stored
                    if d.delta > 0:
                        decisions.append(d)

                    current += 1
                    bar.update(current)

        print()

        r_new_count = r_old_count
        s_new_counts = np.copy(s_old_count)
        g_evolution = []

        # Making the decisions that had the highest positive deltas

        for decision in decisions:
            r_decision = decision.r
            s_decisions = decision.s
            l = decision.line
            c = decision.col

            changes_s = []
            change_r = 0
            values = [pre_proc[l,c] for pre_proc in curr_pre_processed]

            if r[l,c] and not r_decision:
                r_new_count -= 1
                change_r = -1
            
            elif not r[l,c] and r_decision:
                r_new_count += 1
                change_r = 1

            r[l,c] = r_decision

            for i, shell in enumerate(s):
                s_decision = s_decisions[i]

                if shell[l,c] and not s_decision:
                    s_new_counts[i] = s_old_count[i]-1
                    changes_s.append(-1)
                
                elif not shell[l,c] and s_decision:
                    s_new_counts[i] = s_old_count[i]+1
                    changes_s.append(1)

                else:
                    s_new_counts[i] = s_old_count[i]
                    changes_s.append(0)

                shell[l,c] = s_decision

            r_intensity = compute_new_mean_r(change_r, r_old_intensities, values, r_old_count)
            s_intensity = compute_new_mean_s(changes_s, s_old_intensities, values, s_old_count)
            g_new = compute_G(r_new_count, s_new_counts, r_intensity, s_intensity)
            g_evolution.append(g_new)

            r_old_count = np.copy(r_new_count)
            s_old_count = np.copy(s_new_counts)
            r_old_intensities = np.copy(r_intensity)
            s_old_intensities = np.copy(s_intensity)

        # Plotting G evolution
        div = int(len(g_evolution) / 50)
        print(len(g_evolution))
        inds_to_plot = np.linspace(0,len(g_evolution)-1,div,dtype=int)
        g_to_plot = [g_evolution[i] for i in inds_to_plot]
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(g_to_plot)), g_to_plot)
        ax.grid()
        ax.set_title(f'Year {year} - Convergence of clustering criterion G for Upwelling Clump {upwelling_clump+1}', size=11)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('G values')
        plt.xticks(np.linspace(0, len(g_to_plot)-1, 5), np.linspace(0, len(g_to_plot)-1, 5,dtype=int))
        ax.set_yticks(np.linspace(np.min(g_evolution), np.max(g_evolution), 10))
        fig.tight_layout()
        plt.savefig(Path(f'{core_shell_clusters_path}/{starting_map+1}_{ending_map+1}_g_evolution.jpeg'))
        plt.close()

        g_values.append(g_to_plot)

        # Computing the final Core-Shell Cluster, differentiating core from shells

        r_ones_inds = r == 1
        final_result = np.zeros((r.shape[0], r.shape[1]), dtype=int)
        final_result = final_result | r

        for i, shell in enumerate(s):
            final_result = final_result | shell
            s_ones_inds = shell.astype(int) == 1
            core_shell_to_plot = np.zeros((r.shape[0], r.shape[1]))
            core_shell_to_plot[r_ones_inds] = 1
            core_shell_to_plot[s_ones_inds] = 2
            core_shell_to_plot[land_inds] = 3

            cmap = LinearSegmentedColormap.from_list("br", ["dodgerblue", "orange", "green", "white"], N=4)
            plot_image(img=core_shell_to_plot, 
                    title=f'Core-Shell Cluster {i+1} - Upwelling Clump {upwelling_clump+1}',
                    cmap=cmap,
                    show_cbar=True,
                    cbar_labels=['Non\nUpwelling', 'Core', 'Shell', 'Nan'],
                    cbar_ticks=[0.4, 1.1, 1.85, 2.6],
                    min_lat=min_lat, 
                    max_lat=max_lat, 
                    min_long=min_long, 
                    max_long=max_long,
                    fig_size=(figsize_sst[0]+1, figsize_sst[1]),
                    path_to_save=core_shell_clusters_path,
                    save_name=f'clump_{upwelling_clump+1}_core_shell_{i+1}')

        # Computing the final cores' and shells' intensities

        r_intensity = [np.nanmean(pre_proc[r_bool]) for pre_proc in curr_pre_processed]
        s_intensity = [np.nanmean(pre_proc[s_bool[i]]) for i, pre_proc in enumerate(curr_pre_processed)]

        # Saving Core-Shell Cluster info

        to_save = {'upwelling_clump': f'{starting_map}-{ending_map}', 
                'core': r, 
                'shells': s,
                'core_intensities': r_intensity,
                'shells_intensities': s_intensity,
                'g_values': g_to_plot}

        savemat(Path(f'{core_shell_clusters_path}/core_shell_cluster_{upwelling_clump+1}.mat'), to_save)

        print(f'Core`s intensities: {r_intensity}')
        print(f'Shell`s intensities: {s_intensity}')
        print(f'Number of decisions: {len(decisions)}')
        elapsed_time = np.round((time.time() - start_time)/60,2)
        print(f'Elapsed time: {elapsed_time} mins')

        for r_int in r_intensity:
            cores_intensities.append(r_int)

        for s_int in s_intensity:
            shells_intensities.append(s_int)

        kulczynski_multi_values.append(kulczynski_multi(final_result, curr_outputs_sstsec))

        ari_scores = []
        final_result = np.array(final_result).flatten()

        for i, output in enumerate(curr_outputs_sstsec):
            output_aux = np.array(output)
            output_aux[output_aux == 2] = 0
            output_aux = output_aux.flatten()
            conf_matrix_ = pair_confusion_matrix(final_result, output_aux)
            (tn, fp), (fn, tp) = conf_matrix_/conf_matrix_.sum()
            ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
            ari_scores.append(ari)
            aris.append(ari)

            shell = s[i]
            core_shell_instant = r | shell
            sstsec_result = np.copy(output)
            sstsec_result[sstsec_result != 1] = 0

            ks = kulczynski(sstsec_result,core_shell_instant)
            simple_kulczynski_values.append(ks)

        # Plotting ARI scores
        fig, ax = plt.subplots()
        ax.plot(np.arange(starting_map+1, ending_map+2), ari_scores)
        ax.grid()
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f'Year {year} - ARI Scores for Upwelling Clump {upwelling_clump+1}')
        ax.set_xlabel('Map number')
        ax.set_ylabel('ARI')
        ax.set_yticks(np.linspace(np.min(ari_scores), np.max(ari_scores), 10))
        fig.tight_layout()
        plt.savefig(Path(f'{core_shell_clusters_path}/{starting_map+1}_{ending_map+1}_ari_scores.jpeg'))
        plt.close()

        times.append(f'{starting_map}-{ending_map}')
        clumps_run_times.append(elapsed_time)

    file = open(Path(f'{core_shell_clusters_path}/final_results.txt'), 'w')
    file.write('Upwelling Clumps:\n')
    for t in times:
        t_split = t.split('-')
        t_start = int(t_split[0])+1
        t_end = int(t_split[1])+1
        file.write(f'{t_start}-{t_end}\n')

    file.write('\n')
    file.write('Core\'s and shell\'s intensities:\n')

    for i, core_int in enumerate(cores_intensities):
        shell_int = shells_intensities[i]
        file.write(f't={i+1}: core: {core_int}, shell: {shell_int}\n')

    file.write('\n')
    file.write('ARI\'s:\n')

    for i,ari in enumerate(aris):
        file.write(f't={i+1}: ari: {ari}\n')

    file.write('\n')
    file.write(f'Average ARI: {np.mean(aris)}\n')
    file.write('\n')
    file.write('WKS similarities between core shell cluster and its T sstsec maps:\n')

    for i,wks in enumerate(kulczynski_multi_values):
        file.write(f'P={i+1}: wks: {wks}\n')

    file.write('\n')
    file.write(f'Average WKS: {np.mean(kulczynski_multi_values)}\n')
    file.write('\n')
    file.write('KS similarities between core shell cluster and its correspondent sstsec map:\n')

    for i,ks in enumerate(simple_kulczynski_values):
        file.write(f'i={i+1}: ks: {ks}\n')

    file.write('\n')
    file.write(f'Average KS: {np.mean(simple_kulczynski_values)}\n')

    file.write('\n')
    file.write(f'Time per clump:\n')

    for i,t in enumerate(clumps_run_times):
        file.write(f'P={i+1}: {t} mins\n')

    file.close()

    fig, ax = plt.subplots()
    min_g = 9999999
    max_g = 0
    max_len = 0

    for i,g in enumerate(g_values):
        ax.plot(np.arange(len(g)), g, label=f'Upwelling Clump {i+1}')
        min_g_temp = min(g)
        max_g_temp = max(g)

        if min_g_temp < min_g:
            min_g = min_g_temp

        if max_g_temp > max_g:
            max_g = max_g_temp

        if len(g) > max_len:
            max_len = len(g)

    ax.grid()
    ax.set_title(f'Year {year} - Convergence of clustering criterion G', size=11)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('G values')
    plt.xticks(np.linspace(0, max_len, 8), np.linspace(0, max_len, 8,dtype=int))
    ax.set_yticks(np.linspace(min_g, max_g, 10))
    ax.legend()
    fig.tight_layout()
    plt.savefig(Path(f'{core_shell_clusters_path}/g_evolution.jpeg'))
    plt.close()