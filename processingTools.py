import numpy as np

def get_4_neighbours(image, pixel_line, pixel_col, to_add, exclude, ignore=set()):
    img_height = image.shape[0]
    img_width = image.shape[1]

    neigh_l_col = pixel_col - 1
    if pixel_col > 0:
        coords_l = (pixel_line, neigh_l_col)
        temp_l = image[pixel_line, neigh_l_col]
        if not np.isnan(temp_l) and coords_l not in exclude and coords_l not in ignore:
            to_add.add(coords_l)

    neigh_t_line = pixel_line - 1
    if pixel_line > 0:
        coords_t = (neigh_t_line, pixel_col)
        temp_t = image[neigh_t_line, pixel_col]
        if not np.isnan(temp_t) and coords_t not in exclude and coords_t not in ignore:
            to_add.add(coords_t)

    neigh_r_col = pixel_col + 1
    if neigh_r_col < img_width:
        coords_r = (pixel_line, neigh_r_col)
        temp_r = image[pixel_line, neigh_r_col]
        if not np.isnan(temp_r) and coords_r not in exclude and coords_r not in ignore:
            to_add.add(coords_r)

    neigh_b_line = pixel_line + 1
    if neigh_b_line < img_height:
        coords_b = (neigh_b_line, pixel_col)
        temp_b = image[neigh_b_line, pixel_col]
        if not np.isnan(temp_b) and coords_b not in exclude and coords_b not in ignore:
            to_add.add(coords_b)

def get_8_neighbours(image, pixel_line, pixel_col, to_add, exclude, ignore=set()):
    img_height = image.shape[0]
    img_width = image.shape[1]

    get_4_neighbours(image, pixel_line, pixel_col, to_add, exclude, ignore)

    neigh_t_l_col = pixel_col - 1
    neigh_t_l_line = pixel_line - 1
    if pixel_col > 0 and pixel_line > 0:
        coords_t_l = (neigh_t_l_line, neigh_t_l_col)
        temp_t_l = image[neigh_t_l_line, neigh_t_l_col]
        if not np.isnan(temp_t_l) and coords_t_l not in exclude and coords_t_l not in ignore:
            to_add.add(coords_t_l)

    neigh_t_r_col = pixel_col + 1
    neigh_t_r_line = pixel_line - 1
    if pixel_line > 0 and pixel_col < img_width-1:
        coords_t_r = (neigh_t_r_line, neigh_t_r_col)
        temp_t_r = image[neigh_t_r_line, neigh_t_r_col]
        if not np.isnan(temp_t_r) and coords_t_r not in exclude and coords_t_r not in ignore:
            to_add.add(coords_t_r)

    neigh_b_r_col = pixel_col + 1
    neigh_b_r_line = pixel_line + 1
    if pixel_line < img_height-1 and pixel_col < img_width-1:
        coords_b_r = (neigh_b_r_line, neigh_b_r_col)
        temp_b_r = image[neigh_b_r_line, neigh_b_r_col]
        if not np.isnan(temp_b_r) and coords_b_r not in exclude and coords_b_r not in ignore:
            to_add.add(coords_b_r)

    neigh_b_l_col = pixel_col - 1
    neigh_b_l_line = pixel_line + 1
    if pixel_line < img_height-1 and pixel_col > 0:
        coords_b_l = (neigh_b_l_line, neigh_b_l_col)
        temp_b_l = image[neigh_b_l_line, neigh_b_l_col]
        if not np.isnan(temp_b_l) and coords_b_l not in exclude and coords_b_l not in ignore:
            to_add.add(coords_b_l)

def clean_img(img):
    to_clean = []

    for line in range(0, img.shape[0], 1):
        for col in range(img.shape[1]-1, -1, -1):
            if not np.isnan(img[line, col]) and (line, col) not in to_clean:
                is_valid, neighs = in_big_area(img, line, col)

                if not is_valid:
                    for (line,col) in neighs:
                        to_clean.append((line,col))
                else:
                    break

    for (line, col) in to_clean:
        img[line, col] = np.nan

    #print('Cleaned {} pixels'.format(len(to_clean)))

def in_big_area(img, seed_line, seed_col):
        neighbours = set()
        boundary = {(seed_line, seed_col)}
        sec = set()

        while len(boundary) > 0:
            for (line, col) in boundary:
                get_4_neighbours(img, line, col, sec, neighbours)
                neighbours.add((line, col))
        
                if len(neighbours) > 200:
                    return True, []

            boundary = sec.copy()
            sec.clear()

        return False, neighbours

def change_temp_range(img, old_min, old_max, new_min, new_max):
    for line in np.arange(0, img.shape[0], 1):
        for col in np.arange(0, img.shape[1], 1):
            temp = img[line,col]
            if not np.isnan(temp):
                img[line,col] = (((temp - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

#get_grd_multiple('./ImagesPortugal2021/imgs_grd', './ImagesPortugal2021', False)
#get_mat_images('./ImagesMorocco/mat')
#get_dat_images('./CoastLines')