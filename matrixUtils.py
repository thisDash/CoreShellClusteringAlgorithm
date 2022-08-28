import numpy as np

# Computes the frontier of a given binary matrix
# Returns the computed frontier
def get_frontier(region=np.empty((0,0))):
    frontier = np.zeros((region.shape[0], region.shape[1]))
    for l, line in enumerate(region):
        for c, val in enumerate(line):
            if val == 1 and ((c < region.shape[1]-1 and region[l,c+1] != 1) or (c > 0 and region[l,c-1] != 1) or (l > 0 and region[l-1,c] != 1) or (l < region.shape[0]-1 and region[l+1,c] != 1)):
                frontier[l,c] = 1

    return frontier