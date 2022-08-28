from pathlib import Path
import numpy as np
import scipy
import os
from experimentUtils import natural_keys
from anomalousPattern import AnomalousCluster
from copy import deepcopy

class UpwellingClumpsBuilder:

    def __init__(self, segmentations_path='', original_averages_path='', segmentation_files=[], original_files=[]):
        if segmentations_path:
            self.segmentations = []
            dirs = os.listdir(segmentations_path)
            dirs.sort(key=natural_keys)

            for filename in dirs:
                file_path=Path(f'{segmentations_path}/{filename}')
                _, ext = os.path.splitext(filename)

                if ext == '.mat':
                    img_sci = scipy.io.loadmat(file_path)
                    aux = np.array(img_sci['imagem'])
                    img = aux.astype(int)
                    self.segmentations.append(img)

        elif len(segmentation_files) != 0:
            self.segmentations = segmentation_files

        else:
            print('No segmentation path nor files have been provided...')
            exit()

        if original_averages_path:
            self.originals = []
            dirs = os.listdir(original_averages_path)
            dirs.sort(key=natural_keys)

            for filename in dirs:
                file_path=Path(f'{original_averages_path}/{filename}')
                _, ext = os.path.splitext(filename)

                if ext == '.mat':
                    img_sci = scipy.io.loadmat(file_path)
                    img = np.array(img_sci['imagem'])
                    self.originals.append(img)

        elif len(original_files) != 0:
            self.originals = original_files

        else:
            print('No original path nor files have been provided...')
            exit()

        if len(self.segmentations) != len(self.originals):
            print('The number of segmentations and originals are not the same')
            exit()
        
        self.segmentations_path = segmentations_path
        self.original_averages_path = original_averages_path

    def get_features(self, min_lat, max_lat):
        areas_total = []
        temps_total = []

        north_lats = []
        south_lats = []

        latitudes = np.arange(min_lat,max_lat,1/self.segmentations[0].shape[0])

        for segmentation, original in zip(self.segmentations, self.originals):
            non_one_inds = segmentation != 1
            segmentation[non_one_inds] = 0

            south_start_ind = 0
            north_start_ind = segmentation.shape[0]-1

            for i,line in enumerate(segmentation):
                if 1 in line:
                    south_start_ind = i
                    break

            for l in range(segmentation.shape[0]-1, -1, -1):
                if 1 in segmentation[l,:]:
                    north_start_ind = l
                    break

            north_lats.append(latitudes[north_start_ind])
            south_lats.append(latitudes[south_start_ind])
            
            areas_total.append(np.count_nonzero(segmentation)*4)
            temps_total.append(np.nanmean(original[segmentation]))
                
        print(north_lats)
        print(south_lats)
        print()
        areas_total = (areas_total - np.mean(areas_total)) / (np.max(areas_total)-np.min(areas_total))
        temps_total = (temps_total - np.mean(temps_total)) / (np.max(temps_total)-np.min(temps_total))

        if np.max(north_lats) == np.min(north_lats):
            north_lats = north_lats - np.max(north_lats)
        else:
            north_lats = (north_lats - np.mean(north_lats)) / (np.max(north_lats)-np.min(north_lats))
            
        
        if np.max(south_lats) == np.min(south_lats):
            south_lats -= np.max(south_lats)
        else:
            south_lats = (south_lats - np.mean(south_lats)) / (np.max(south_lats)-np.min(south_lats))

        timesteps = np.arange(0,len(areas_total))
        timesteps = (timesteps - np.mean(timesteps)) / (np.max(timesteps)-np.min(timesteps))
    
        return timesteps, areas_total, temps_total, north_lats, south_lats

    def get_clumpss(self, min_partition_size=3, min_lat=None, max_lat=None, use_south_lats=True, use_north_lats=True):
        if min_lat is None or max_lat is None:
            print('get_clumps(): Coordinates error...')
            exit()

        timesteps, areas_total, temps_total, north_lats, south_lats = self.get_features(min_lat=min_lat, max_lat=max_lat)
        features_array = [timesteps, areas_total, temps_total]

        if use_south_lats:
            features_array.append(south_lats)

        if use_north_lats:
            features_array.append(north_lats)
            
        features = np.c_[timesteps, areas_total, temps_total, north_lats, south_lats]
        anom_patterns = AnomalousCluster(normalization=False,threshold=0)
        cents, standard_data, anom_clusters, _ = anom_patterns.fit_transform(features)
        
        total_scatter_data = 0

        for line in standard_data:
            for val in line:
                total_scatter_data += (val ** 2)

        contributions = []
        for i, cent in enumerate(cents):
            rel_cont = len(anom_clusters[i]) * (np.sum(np.power(cent,2)) * 100 / total_scatter_data)
            contributions.append(rel_cont)
        
        changed = True

        while changed:
            changed = False
            for i_anom, anom_cluster in enumerate(anom_clusters):
                if len(anom_cluster) < min_partition_size:
                    candidates = []

                    for i, master_cluster in enumerate(anom_clusters):
                        if master_cluster != anom_cluster and (anom_cluster[0] == (master_cluster[-1] + 1)) or (anom_cluster[-1] == (master_cluster[0] - 1)):
                            candidates.append((i,master_cluster))

                    if len(candidates) > 0:
                        changed = True
                        min_contribution = contributions[candidates[0][0]]
                        master = candidates[0][1]

                        for c, candidate in candidates:
                            if contributions[c] < min_contribution:
                                min_contribution = contributions[c]
                                master = candidate

                        master += anom_cluster
                        master.sort()
                        anom_clusters.remove(anom_cluster)
                        contributions.remove(contributions[i_anom])
                        break

        return anom_clusters