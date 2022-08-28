import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import numpy as np
import sys, os

from stsec import STSEC
from numpy.core.numeric import NaN
from matplotlib.colors import LinearSegmentedColormap
from regionGrowing import RegionGrowing

from fileUtils import get_grd_file

class SSTSEC(STSEC):

    def __init__(self, file_path, file_name, file_format='', to_log=False, file=np.empty((0,0))):
        super().__init__(file_path, file_name, file_format, to_log, file)

        if len(file) != 0:
            self.otsu_img = np.copy(file)

        else:
            if file_format == '.mat':
                self.otsu_img = np.array(scipy.io.loadmat(file_path)['imagem'])

            elif file_format == '.grd':
                self.otsu_img, _, _ = get_grd_file(file_path, file_name)

    def __get_mean_temp(self, region):
        sum_temps = 0

        for (line, col) in region:
            sum_temps += self.image[line, col]

        return sum_temps / len(region)

    def __get_min_temp(self, region):
        min_temp = 50

        for (line, col) in region:
            temp = self.image[line, col]
            if temp < min_temp:
                min_temp = temp

        return min_temp

    def __get_otsu_thresh(self):
        sec = []
        max_temp = 0

        for line_data in self.otsu_img:
            for temp in line_data:
                if not np.isnan(temp):
                    sec.append(temp)
                    if temp > max_temp:
                        max_temp = temp

        sec = np.array(sec)
        sec = sec / max_temp * 255
        sec_img_mean = np.mean(sec)
        thresh = self.__otsu(sec, sec_img_mean)
        thresh = thresh / 255 * max_temp
        thresh -= np.nanmean(self.otsu_img)
        return thresh

    def __otsu(self, gray, img_mean):
        pixel_number = len(gray)
        mean_weight = 1.0/pixel_number
        his, bins = np.histogram(gray, np.arange(0,257))
        final_thresh = -1.0
        final_value = -1.0
        intensity_arr = np.arange(256)
        for t in bins[1:-1]:
            pcb = np.sum(his[:t])
            pcf = np.sum(his[t:])
            Wb = pcb * mean_weight
            Wf = pcf * mean_weight

            mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
            muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value and t >= img_mean:
                final_thresh = t
                final_value = value
                
        return final_thresh

    def segment(self):
        otsu_thresh = self.__get_otsu_thresh()

        output, region, total_its = super().segment()
        mean_first_region = self.__get_mean_temp(region)

        srg = RegionGrowing(self.image)
        regions_grown = 1        
        
        while True:
            new_seed = self._STSEC__get_next_seed(region)

            if new_seed is NaN:
                #print('No seed found, ending algorithm')
                break

            srg.set_new_seed(new_seed)
            new_region, its = srg.grow_region(region)
            total_its += its

            if len(new_region) < 225:
                self.excluded_seeds.add(new_seed.coords)

            else:
                min_temp = self.__get_min_temp(new_region)
                dif = mean_first_region - min_temp

                if dif <= otsu_thresh:
                    break

                for (line, col) in new_region:
                    output[line, col] = 1
                    region.add((line, col))

                regions_grown += 1

        return output, region, total_its
                
    #Uses the region obtained to create an image with the segmentation
    def save_image(self, image, path_to_save):
        x_axis = np.arange(0, self.img_width, 5)
        y_axis = np.linspace(0, self.img_height, 5)
        cmap = LinearSegmentedColormap.from_list("br", ["white", "black"], N=2)
        ax = sns.heatmap(image, cmap=cmap)
        #ax.invert_yaxis()
        plt.yticks(y_axis, y_axis)
        plt.xticks(x_axis, x_axis)
        plt.savefig('{}/{}_sstsec.jpg'.format(path_to_save, self.file_name))
        plt.close()