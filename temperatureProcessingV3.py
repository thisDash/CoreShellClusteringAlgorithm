import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experimentUtils import get_files_from_mat, get_frontier, get_land_pixels
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
from scipy.io import savemat
from sklearn.linear_model import LinearRegression
from plotters import add_heatmap, plot_image
from pathlib import Path

# Temperature preprocessing for the Portuguese coastline
class TemperaturePreprocessingV3:

     def __init__(self, path_to_process='', file=[], file_name='', files=[], files_names=[], min_lat=None, max_lat=None, min_long=None, max_long=None):
          if min_lat is None or max_lat is None or min_long is None or max_long is None or min_lat == max_lat or min_long == max_long:
               print("Coordinates aren't well specified...")
               exit()

          self.min_lat = min_lat
          self.max_lat = max_lat
          self.min_long = min_long
          self.max_long = max_long

          if path_to_process:
               self.files, self.names = get_files_from_mat(path_to_process)

          elif len(file) != 0:
               if not file_name:
                    print('No name provided for the file')
                    exit()

               self.files = [file]
               self.names = [file_name]

          elif len(files) > 0:
               if len(files) != len(files_names):
                    print('Files and file names counts are different')

               self.files = files
               self.names = files_names

          else:
               print('No files or names provided')
               exit()

     # Gets perpendicular points until first horizontal is found
     def get_perp_points(self, land_matrix, coast_pixels, pixels_range, n_points):
          print('##########')
          print()
          print('Finding perpendicular points...')
          print()
          perp_points_set = []
          equations_set = []

          #First n_points/2 vertical lines
          for (l,c) in coast_pixels[:int(n_points/2)]:
               aux_set = []
               nans_before  = np.count_nonzero(land_matrix[0:l,c])
               nans_after = np.count_nonzero(land_matrix[l+1:land_matrix.shape[0],c])

               if nans_before > nans_after:
                    line_range = np.arange(l+1,land_matrix.shape[0],1)

               else:
                    line_range = np.arange(0,l,1)

               for line in line_range:
                    if land_matrix[line,c] == 0:
                         aux_set.append((line,c))

               perp_points_set.append(aux_set)
               equations_set.append((-1,c))

          for i in pixels_range:
               points = coast_pixels[i:i+n_points]
               first_point = points[0]
               last_point = points[-1]
               mid_point = points[int(n_points/2)]
               aux_set = []

               #Vertical perpendicular
               if last_point[0] == first_point[0]:
                    nans_before  = np.count_nonzero(land_matrix[mid_point[0]-5:mid_point[0],mid_point[1]])
                    nans_after = np.count_nonzero(land_matrix[mid_point[0]+1:mid_point[0]+6,mid_point[1]])

                    if nans_before > nans_after:
                         line_range = np.arange(mid_point[0]+1,land_matrix.shape[0],1)

                    else:
                         line_range = np.arange(0,mid_point[0],1)

                    for line in line_range:
                         if land_matrix[line,mid_point[1]] == 0:
                              aux_set.append((line,mid_point[1]))

                    equations_set.append((-1,mid_point[1]))

               #Horizontal perpendicular
               elif last_point[1] >= first_point[1]:
                    first_horizontal_line = mid_point[0]-1
                    break

               else:
                    #Xs
                    cols = np.array([points[0][1], points[-1][1]]).reshape((-1,1))
                    #Ys
                    lines = np.array([points[0][0], points[-1][0]])

                    model = LinearRegression().fit(cols,lines)

                    #Perpendicular params
                    m = -1/model.coef_
                    b = (mid_point[0] - (m * mid_point[1]))
                    #Get all points before mid point

                    cols_test_before = np.arange(mid_point[1]-1,-1,-1)
                    pixels_before = []

                    for col in cols_test_before:
                         line = int(m * col + b)
                         if line > 0 and line < land_matrix.shape[0]:
                              pixels_before.append((line,col))
                         else:
                              break

                    # If no pixels before, treat it as a vertical
                    if len(pixels_before) == 0:
                         print(i)
                         nans_before  = np.count_nonzero(land_matrix[mid_point[0]-5:mid_point[0],mid_point[1]])
                         nans_after = np.count_nonzero(land_matrix[mid_point[0]+1:mid_point[0]+6,mid_point[1]])

                         if nans_before > nans_after:
                              line_range = np.arange(mid_point[0]+1,land_matrix.shape[0],1)

                         else:
                              line_range = np.arange(0,mid_point[0],1)

                         for line in line_range:
                              if land_matrix[line,mid_point[1]] == 0:
                                   aux_set.append((line,mid_point[1]))

                         equations_set.append((-1,mid_point[1]))
                         perp_points_set.append(aux_set)
                         continue

                    elif pixels_before[0][0] < mid_point[0]:
                         lines_test_before = np.arange(mid_point[0]-1,-1,-1)

                         for line in lines_test_before:
                              col = int((line - b) / m)
                              if col >= 0 and col <= mid_point[1] and (line,col) not in pixels_before:
                                   pixels_before.append((line,col))

                    else:
                         lines_test_before = np.arange(mid_point[0]+1,land_matrix.shape[0],1)

                         for line in lines_test_before:
                              col = int((line - b) / m)
                              if col >= 0 and col <= mid_point[1] and (line,col) not in pixels_before:
                                   pixels_before.append((line,col))
                                   
                    nans_before = np.count_nonzero([land_matrix[line,col] for line,col in pixels_before])
                    del cols_test_before
                    del lines_test_before
                    del line
                    del col

                    #Get all points after mid point

                    cols_test_after = np.arange(mid_point[1]+1,land_matrix.shape[1],1)
                    pixels_after = []

                    for col in cols_test_after:
                         line = int(m * col + b)
                         if line > 0 and line < land_matrix.shape[0]:
                              pixels_after.append((line,col))

                         else:
                              break

                    # If no pixels after, treat it as a vertical
                    if len(pixels_after) == 0:
                         nans_before  = np.count_nonzero(land_matrix[mid_point[0]-5:mid_point[0],mid_point[1]])
                         nans_after = np.count_nonzero(land_matrix[mid_point[0]+1:mid_point[0]+6,mid_point[1]])

                         if nans_before > nans_after:
                              line_range = np.arange(mid_point[0]+1,land_matrix.shape[0],1)

                         else:
                              line_range = np.arange(0,mid_point[0],1)

                         for line in line_range:
                              if land_matrix[line,mid_point[1]] == 0:
                                   aux_set.append((line,mid_point[1]))

                         equations_set.append((-1,mid_point[1]))
                         perp_points_set.append(aux_set)
                         continue

                    elif pixels_after[0][0] < mid_point[0]:
                         lines_test_after = np.arange(mid_point[0]-1,-1,-1)

                         for line in lines_test_after:
                              col = int((line - b) / m)
                              if col >= mid_point[1] and col < land_matrix.shape[1] and (line,col) not in pixels_after:
                                   pixels_after.append((line,col))

                    else:
                         lines_test_after = np.arange(mid_point[0]+1,land_matrix.shape[0],1)

                         for line in lines_test_after:
                              col = int((line - b) / m)
                              if col >= mid_point[1] and col < land_matrix.shape[1] and (line,col) not in pixels_after:
                                   pixels_after.append((line,col))

                    nans_after = np.count_nonzero([land_matrix[line,col] for line,col in pixels_after])

                    del cols_test_after
                    del lines_test_after
                    del line
                    del col

                    equations_set.append((m,b))

                    #Gets water pixels, set with less NaNs
                    if nans_before > nans_after:
                         for (l,c) in pixels_after:
                              if land_matrix[l,c] == 0:
                                   aux_set.append((l,c))

                    else:
                         for (l,c) in pixels_before:
                              if land_matrix[l,c] == 0:
                                   aux_set.append((l,c))

               perp_points_set.append(aux_set)

          return perp_points_set, first_horizontal_line, equations_set

     # Atributes missing points to closest equations
     def assign_missing_points(self, missing_points, equations, points_sets):
          for (l,c) in missing_points:
               smallest_dif = 9999
               smallest_ind = 0

               for i,(slope,intercept) in enumerate(equations):

                    #Horizontal line
                    if slope == 0:
                         dif = abs(l - intercept)

                    #Vertical line
                    elif slope == -1:
                         continue

                    else:
                         dif = abs(l - slope*c - intercept)

                    if dif < smallest_dif:
                         smallest_dif = dif
                         smallest_ind = i

               points_sets[smallest_ind].append((l,c))

     # Gets coastline in order, from top right to bottom right
     def get_ordered_pixels(self, land):
          #Finds the top right pixel of the land pixels found
          #TODO Change how this is computed?
          for l, pixel in enumerate(land[:,-1]):
               if pixel == 1:
                    curr_points = [(l,land.shape[1]-1)]
                    break

          coastline = get_frontier(land)
          coast_pixels = []
          coast_matrix = np.zeros((land.shape[0], land.shape[1]))
          coast_matrix[land == 1] = np.nan

          #Finds the coastline in order, pixel by pixel in a RG way
          while (len(curr_points) > 0):
               curr_point = curr_points[0]
               curr_points.remove(curr_point)
               coast_pixels.append(curr_point)

               l = curr_point[0]
               c = curr_point[1]

               coast_matrix[l,c] = 1

               try:
                    if coastline[l-1,c] == 1 and (l-1,c) not in coast_pixels and (l-1,c) not in curr_points:
                         curr_points.append((l-1,c))
               except:
                    pass

               try:
                    if coastline[l+1,c] == 1 and (l+1,c) not in coast_pixels and (l+1,c) not in curr_points:
                         curr_points.append((l+1,c))
               except:
                    pass

               try:
                    if coastline[l,c-1] == 1 and (l,c-1) not in coast_pixels and (l,c-1) not in curr_points:
                         curr_points.append((l,c-1))
               except:
                    pass

               try:
                    if coastline[l,c+1] == 1 and (l,c+1) not in coast_pixels and (l,c+1) not in curr_points:
                         curr_points.append((l,c+1))
               except:
                    pass

               try:
                    if coastline[l-1,c-1] == 1 and (l-1,c-1) not in coast_pixels and (l-1,c-1) not in curr_points:
                         curr_points.append((l-1,c-1))
               except:
                    pass

               try:
                    if coastline[l-1,c+1] == 1 and (l-1,c+1) not in coast_pixels and (l-1,c+1) not in curr_points:
                         curr_points.append((l-1,c+1))
               except:
                    pass

               try:
                    if coastline[l+1,c-1] == 1 and (l+1,c-1) not in coast_pixels and (l+1,c-1) not in curr_points:
                         curr_points.append((l+1,c-1))
               except:
                    pass

               try:
                    if coastline[l+1,c+1] == 1 and (l+1,c+1) not in coast_pixels and (l+1,c+1) not in curr_points:
                         curr_points.append((l+1,c+1))
               except:
                    pass

          return coast_pixels

     def preprocess(self, path_to_save_img='', path_to_save_mat='', path_to_save_land='', to_animate=False, n_points=50, fig_size=None):
          results = []

          if path_to_save_img:
               Path(path_to_save_img).mkdir(parents=True, exist_ok=True)

               if fig_size is None:
                    print('Fig size not specified...')
                    exit()
          
          if path_to_save_mat:
               Path(path_to_save_mat).mkdir(parents=True, exist_ok=True)

          if path_to_save_land:
               Path(path_to_save_land).mkdir(parents=True, exist_ok=True)
          
          for i, img in enumerate(self.files):
               print('##########')
               print()
               print(f'Preprocessing instant {i+1}...')
               print()
               
               name = self.names[i]
               land = get_land_pixels(img)

               if path_to_save_land:
                    to_save = {'imagem': land}
                    savemat(f'{path_to_save_land}/{name}.mat', to_save)

               print('##########')
               print()
               print('Finding the coastline pixels in order...')
               print()

               coast_pixels = self.get_ordered_pixels(land)

               #For the north analysis
               rev_coast_pixels = coast_pixels.copy()
               rev_coast_pixels.reverse()

               print('##########')
               print()
               print('Finding the north and south perpendicular equations...')
               print()

               south_points_sets, first_horizontal_line, south_equations = self.get_perp_points(land,coast_pixels,np.arange(0,len(coast_pixels)-n_points,1), n_points)
               north_points_sets, last_horizontal_line, north_equations = self.get_perp_points(land,rev_coast_pixels,np.arange(0,len(coast_pixels)-n_points,1), n_points)

               south_matrix = land[:first_horizontal_line+1,:].copy()

               for set_points in south_points_sets:
                    for (l,c) in set_points:
                         south_matrix[l,c] = 1

               missing_lines, missing_cols = np.where(south_matrix == 0)
               self.assign_missing_points(list(zip(missing_lines,missing_cols)),south_equations, south_points_sets)

               north_matrix = land[last_horizontal_line:,:].copy()

               for i, set_points in enumerate(north_points_sets):
                    for (l,c) in set_points:
                         north_matrix[l-last_horizontal_line,c] = 1

               missing_lines, missing_cols = np.where(north_matrix == 0)
               missing_lines = np.add(missing_lines,last_horizontal_line)
               self.assign_missing_points(list(zip(missing_lines,missing_cols)),north_equations, north_points_sets)
               north_points_sets.reverse()

               print('##########')
               print()
               print('Building final equations...')
               print()

               #Build final equations array
               final_eqs = []

               final_eqs += south_points_sets

               for line in np.arange(first_horizontal_line+1,last_horizontal_line,1):
                    eq = []

                    for col in np.arange(0,img.shape[1]):
                         if land[line,col] == 0:
                              eq.append((line,col))

                    final_eqs.append(eq)

               final_eqs += north_points_sets

               final_img = np.zeros(img.shape)

               print('##########')
               print()
               print('Doing final temperature preprocessing...')
               print()

               temp_range = np.abs(np.nanmax(img) - np.nanmin(img))

               for i, perp_set in enumerate(final_eqs):
                    avg_temp = np.nanmean([img[line,col] for line,col in perp_set])

                    for line,col in perp_set:
                         if img[line,col] > (avg_temp - (temp_range * 0.06)):
                              final_img[line,col] = 0

                         else:
                              final_img[line,col] = img[line,col] - avg_temp 
               
               final_img[land == 1] = np.nan

               window_size=5
               filtered_img = final_img.copy()
               offset = int(window_size/2)

               for line, line_data in enumerate(final_img):
                    for col, temp in enumerate((line_data)):
                         if not np.isnan(temp):
                              min_col = col - offset
                              max_col = col + offset
                              min_line = line - offset
                              max_line = line + offset
                              temps = []

                              for line_window in np.arange(min_line, max_line + 1):
                                   if line_window > 0 and line_window < img.shape[0]:
                                        for col_window in np.arange(min_col, max_col + 1):
                                             if col_window > 0 and col_window < img.shape[1]:
                                                  temp = final_img[line_window, col_window]
                                                  if not np.isnan(temp):
                                                       temps.append(temp)

                              filtered_img[line, col] = np.mean(np.array(temps))

               if path_to_save_img:
                    cmap = LinearSegmentedColormap.from_list("br", ["midnightblue", "blue", "cyan", "yellow", "red", "darkred"], N=192)

                    min_val = round(np.nanmin(filtered_img),2)
                    max_val = round(np.nanmax(filtered_img),2)
                    cbar_labels = []
                    for j in np.linspace(min_val, max_val, 10):
                         cbar_labels.append(round(j,2))

                    plot_image(img=filtered_img,
                              cmap=cmap,
                              cbar_labels=cbar_labels,
                              min_lat=self.min_lat,
                              max_lat=self.max_lat,
                              min_long=self.min_long,
                              max_long=self.max_long,
                              fig_size=fig_size,
                              path_to_save=path_to_save_img,
                              save_name=name)
          
               if path_to_save_mat:
                    to_save = {'imagem': filtered_img}
                    savemat(Path(f'{path_to_save_mat}/{name}.mat'), to_save)

               results.append(filtered_img)

          if to_animate and path_to_save_mat:
               to_anim = []
               for points in final_eqs:
                    matrix = np.zeros(img.shape)

                    for (line,col) in points:
                         matrix[line,col] = 1

                    for (line,col) in coast_pixels:
                         matrix[line,col] = 2

                    to_anim.append(matrix)

               print('##########')
               print()
               print('Creating the animation...')
               print()

               fig = plt.figure(figsize=(9,10))
               ax = sns.heatmap(to_anim[0])
               ax.set_aspect('equal', 'box')

               def init():
                    plt.clf()
                    ax = sns.heatmap(to_anim[0], cbar=False)
                    ax.set_aspect('equal', 'box')
                    ax.invert_yaxis()

               def animate(i):
                    plt.clf()
                    print(i)
                    ax = sns.heatmap(to_anim[i])
                    ax.set_aspect('equal', 'box')
                    ax.invert_yaxis()

               anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(to_anim), interval=1000, repeat=True, cache_frame_data=False)

               writergif = animation.PillowWriter(fps=30) 
               anim.save(Path(f'{path_to_save_mat}/perpendiculars_animation.gif'), writer=writergif)

          return results

'''
imgs_path = f'./ImagesTimeSeries/2015/mat/preprocessing_phase_2'
averaged_originals_path = f'./ImagesTimeSeries/2015/mat/april_october_moving_avg_wind_5'
to_animate = False

preprocesser = TemperaturePreprocessingV3(imgs_path)
preprocesser.preprocess('./preprocessV3', './preprocessV3/mat')


img_path = f'./ImagesTimeSeries/2007/mat/preprocessing_phase_2/window_5_n_0.mat'
img = np.array(scipy.io.loadmat(img_path)['imagem'])

preprocesser = TemperaturePreprocessingV3(file=img, file_name='window_5_n_0')
preprocesser.preprocess('./test')
'''