import os, sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join('')))

from matplotlib.colors import LinearSegmentedColormap
from processingTools import get_4_neighbours
from fileUtils import get_grd_file

class CoastlineFinder:

    def __init__(self, file_path='', file_name='', file_format='.mat', file=np.empty((0,0))):
        self.file_name = file_name

        if len(file) != 0:
            self.image = np.copy(file)
        
        elif file_format == '.mat':
            self.image = np.array(scipy.io.loadmat(file_path)['imagem'])

        elif file_format == '.grd':
            self.image, _, _ = get_grd_file(file_path, file_name)

        self.image -= np.nanmean(self.image)
        self.img_height = self.image.shape[0]
        self.img_width = self.image.shape[1]

    #Checks if the point is a valid coastal point
    def __is_valid(self, center_line, center_col, coast_points, window_size=5):
        if not self.__in_big_area(center_line, center_col):
            return False

        offset = int(window_size/2)
        neighs = 0
        for line in range(center_line-offset, center_line+offset+1):
            if line >= 0 and line < self.img_height:
                for col in range(center_col-offset, center_col+offset+1):
                    if col >= 0 and col < self.img_width and (line, col) != (center_line, center_col) and (line, col) in coast_points:
                        neighs += 1
                        if neighs >= 2:
                            return True
            
        return False

    #Checks if the point is in an enclosed cluster of points (noise, small groups between clouds)
    def __in_big_area(self, seed_line, seed_col):
        neighbours = set()
        boundary = {(seed_line, seed_col)}
        sec = set()

        while len(boundary) > 0:
            for (line, col) in boundary:
                get_4_neighbours(self.image, line, col, sec, neighbours)
                neighbours.add((line, col))
        
                if len(neighbours) > 50:
                    return True

            boundary = sec.copy()
            sec.clear()

        return False

    def get_coastline_points(self):
        sec_output = set()
        output_set = set()

        #print(f'Image shape: {self.image.shape}')

        for line in range(self.img_height-1, -1, -1):
            for col in range(self.img_width-1, -1, -1):
                if not np.isnan(self.image[line, col]):
                    if col < self.img_width - 1:
                        sec_output.add((line, col))
                    break
                
        for (line, col) in sec_output:
            if self.__is_valid(line, col, sec_output):
                output_set.add((line, col))

        #print(f'Number of coastline points: {len(output_set)}')

        return output_set

    def get_points_near_coastline(self, coast_points, distance, pixel_res):
        front = coast_points.copy()
        front_prime = set()
        points = set()

        for (seed_line, seed_col) in coast_points:
            
            points.add((seed_line, seed_col))
            get_4_neighbours(self.image, seed_line, seed_col, front, points)

            while len(front) > 0:
                for (line, col) in front:
                    dist = np.sqrt(np.square(seed_line - line) + np.square(seed_col - col)) * pixel_res
                    if dist <= distance:
                        get_4_neighbours(self.image, line, col, front_prime, points)
                        points.add((line, col))
            
                front = front_prime.copy()
                front_prime.clear()

        return points