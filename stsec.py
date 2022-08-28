import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os

from matplotlib.colors import LinearSegmentedColormap
from numpy.core.numeric import NaN
from regionGrowing import RegionGrowing
from coastlineFinder import CoastlineFinder
from processingTools import clean_img

from fileUtils import get_grd_file

class Pixel:

    def __init__(self, coords, temp):
        self.coords = coords
        self.temp = temp

    def __eq__(self, other): 
        if not isinstance(other, Pixel):
            return NotImplemented

        return self.coords == other.coords and self.temp == other.temp

    def __hash__(self):
        return hash((self.coords, self.temp))

class STSEC:

    def __init__(self, file_path, file_name, file_format, to_log=False, file=np.empty((0,0))):
        if len(file) != 0:
            self.image = np.copy(file)

        else:
            self.file_name = file_name
            self.file_path = file_path

            if file_format == '.mat':
                self.image = np.array(scipy.io.loadmat(file_path)['imagem'])

            elif file_format == '.grd':
                self.image, _, _ = get_grd_file(file_path, file_name)
        
        self.excluded_seeds = set()
        clean_img(self.image)

        if to_log:
            #print('Logged')
            self.image = np.log(self.image)

        self.image -= np.nanmean(self.image)
        self.img_height = self.image.shape[0]
        self.img_width = self.image.shape[1]
        cf = CoastlineFinder(file_path, file_name, file_format, file)
        coast_points = cf.get_coastline_points()
        #print('Coast points: {}'.format(len(coast_points)))
        self.seed_candidates = cf.get_points_near_coastline(coast_points, 10, 1.1)
        #print('Seed Candidates: {}'.format(len(self.seed_candidates)))

    #Gets a seed
    def __get_next_seed(self, ignore=set()):
        min_temp = 50
        min_pixel = (0,0)
        changed = False

        for (cand_line, cand_col) in self.seed_candidates:
            temp = self.image[cand_line, cand_col]
            pixel = (cand_line, cand_col)
            if temp < min_temp and pixel not in self.excluded_seeds and pixel not in ignore:
                min_temp = temp
                min_pixel = pixel
                changed = True

        if not changed:
            #print("No seed found near the coast...")
            return NaN

        return Pixel(min_pixel, min_temp)

    #Main method to do the segmentation
    def segment(self):
        initial_seed = self.__get_next_seed()
        output = np.zeros((self.img_height,self.img_width))
        region = set()
        total_its = 0

        if initial_seed is not NaN:
            srg = RegionGrowing(self.image, initial_seed)
            valid_region = False

            while not valid_region:
                region, its = srg.grow_region()

                if len(region) < 225:
                    self.excluded_seeds.add(initial_seed.coords)
                    initial_seed = self.__get_next_seed()
                    srg.set_new_seed(initial_seed)
                else:
                    valid_region = True
                    total_its = its

            for (line, col) in region:
                output[line, col] = 1

        return output, region, total_its

    #Uses the region obtained to create an image with the segmentation
    def save_image(self, image, path_to_save):
        x_axis = np.arange(0, self.img_width, 5)
        y_axis = np.linspace(0, self.img_height, 5)
        cmap = LinearSegmentedColormap.from_list("br", ["blue", "red"], N=2)
        ax = sns.heatmap(image, cmap=cmap)
        #ax.invert_yaxis()
        plt.yticks(y_axis, y_axis)
        plt.xticks(x_axis, x_axis)
        plt.savefig('{}{}_stsec.jpg'.format(path_to_save, self.file_name))
        plt.close()