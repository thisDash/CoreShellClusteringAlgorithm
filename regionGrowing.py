import numpy as np
import sys, os
from numpy.core.numeric import NaN
from processingTools import get_4_neighbours

class RegionGrowing:
    def __init__(self, image, seed=NaN):
        self.image = image
        self.seed = seed
        self.img_height = self.image.shape[0]
        self.img_width = self.image.shape[1]

    def set_image(self, image):
        self.image = image

    def set_new_seed(self, seed):
        self.seed = seed

    def __get_window(self, center_line, center_col, window_size=7):
        offset = int(window_size/2)
        window = np.zeros((window_size, window_size), dtype=tuple)

        for l_window, l_img in enumerate(range(center_line-offset, center_line+offset+1)):
            if l_img >= 0 and l_img < self.img_height:
                for c_window, c_img in enumerate(range(center_col-offset, center_col+offset+1)):
                    if c_img >= 0 and c_img < self.img_width:
                        window[l_window, c_window] = (l_img, c_img)
                    else:
                        window[l_window, c_window] = np.nan

            else:
                for c_window, c_img in enumerate(range(center_col-offset, center_col+offset+1)):
                    window[l_window, c_window] = np.nan
            
        return window

    def __get_similarity_threshold(self, c, center_line, center_col):
        window = self.__get_window(center_line, center_col)
        sum_value = 0
        count = 0

        for line in window:
            for coords in line:
                if coords in c:
                    sum_value += self.image[coords[0], coords[1]]
                    count += 1

        return np.square(sum_value / count)/2

    def __initialize_c(self, ignore):
        c = {self.seed.coords}
        (center_line, center_col) = self.seed.coords
        seed_temp = self.seed.temp
        window = self.__get_window(center_line, center_col)

        for line_data in window:
            for coords in line_data:
                if type(coords) is tuple and coords not in ignore:
                    sim = self.image[coords[0], coords[1]] * seed_temp
                    pi = self.__get_similarity_threshold(c, coords[0], coords[1])

                    if sim >= pi:
                        c.add(coords)

        return c

    def __initialize_f(self, c, ignore):
        f = set()

        for (line, col) in c:
            get_4_neighbours(self.image, line, col, f, c, ignore)
                
        return f

    def grow_region(self, ignore=set()):
        c = self.__initialize_c(ignore)
        f = self.__initialize_f(c, ignore)
        f_prime = set()
        c_prime = set()
        changed = True
        its = 0

        while(changed):
            changed = False
            its += 1
            
            for (bound_pixel_line, bound_pixel_col) in f:
                bound_pixel_temp = self.image[bound_pixel_line, bound_pixel_col]
                window = self.__get_window(bound_pixel_line, bound_pixel_col)
                window_temps = []

                for line in window:
                    for coords in line:                    
                        if type(coords) is tuple and coords in c:
                            window_temps.append(self.image[coords[0], coords[1]])
                        
                sim = bound_pixel_temp * np.nanmean(window_temps)
                pi = self.__get_similarity_threshold(c, bound_pixel_line, bound_pixel_col)
                
                if sim >= pi:
                    changed = True
                    c_prime.add((bound_pixel_line, bound_pixel_col))
                    get_4_neighbours(self.image, bound_pixel_line, bound_pixel_col, f_prime, c, ignore)
                    
            for coords in c_prime:
                c.add(coords)

            c_prime.clear()
            f.clear()
            f = f_prime.copy()
            f_prime.clear()

        return c, its
