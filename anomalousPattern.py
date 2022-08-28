import numpy as np

def center_(x, cluster):
    """ finds the centroid of a cluster
    X - the original data matrix
     cluster - the set with indices of the objects belonging to the cluster
    """
    #number of columns
    mm = x.shape[1]
    centroidC = []
    
    for j in range(mm):
        zz = x[:, j]
        zc = []
        for i in cluster:
            zc.append(zz[i])
        centroidC.append(np.mean(zc))
    return centroidC


def distNorm(x ,remains, ranges, p):
    """ Finds the normalized distances of data points in 'remains' to reference point 'p' 
     X - the original data matrix;
     remains- the set of X-row indices under consideration
     ranges- the vector with ranges of data features 
     p - the data point the distances relate to
     distan- the output column of distances from a to remains """

    #number of columns
    mm = x.shape[1]
    rr = len(remains)
    z = x[remains, :]
    az = np.tile(np.array(p), (rr, 1))
    rz = np.tile(np.array(ranges), (rr, 1))
    dz = (z - az) / rz
    dz = np.array(dz)
    ddz = dz * dz
    if mm > 1:
        di = sum(ddz.T)
    else:
        di = ddz.T
    distan = di
    return distan


def separCluster(x0, remains, ranges, a, b):
    """  Builds a cluster by splitting the points around refernce point 'a' from those around reference point b 
    x0 - data matrix
    remains- the set of X-row indices under consideration
    ranges- the vector with ranges of data features 
    a, b - the reference points
    cluster - set with row indices of the objects belonging to the cluster  
    """
    
    dista = distNorm(x0, remains, ranges, a)
    distb = distNorm(x0, remains, ranges, b)
    clus = np.where(dista < distb)[0]
    cluster = []
    for i in clus:
        cluster.append(remains[i])
    return cluster

def anomalousPattern(x, remains, ranges, centroid, me):
    """ Builds one anomalous cluster based on the algorithm 'Separate/Conquer' (Mirkin, 1999, Machine Learning Journal) 
        X - data matrix,
        remains - set of its row indices (objects) under consideration,
        ranges - normalizing values: the vector with ranges of data features  
        centroid - initial center of the anomalous cluster being build
        me - vector to shift the 0 (origin) to,
        output: cluster - set of row indices in the anomalous cluster, 
        centroid -center of the cluster    """
        
    key = 1
    while key == 1:
        cluster = separCluster(x, remains, ranges, centroid, me)
        if len(cluster) != 0:
            newcenter = center_(x, cluster)
        if  len([i for i, j in zip(centroid, newcenter) if i == j]) != len(centroid):
            centroid = newcenter
        else:
            key = 0
    return (cluster, centroid)

def dist(x, remains, ranges, p):
    """ Calculates the normalized distances of data points in 'remains' to reference point 'p'   
        X - data matrix,
        remains - set of its row indices (objects) under consideration,
        ranges - normalizing values: the vector with ranges of data features  
    
       distan - the calculated normalized distances
    """

    #number of columns
    mm = x.shape[1]
    rr = len(remains)
    distan = np.zeros((rr,1))    
    for j in range(mm):
        z = x[:, j]
        z = z.reshape((-1,1))
        zz = z[remains]
        y = zz - p[j]
        y = y / ranges[j]
        y = np.array(y)
        yy = y * y
        distan = distan + yy
    return distan    

class AnomalousCluster:
    
    def __init__(self,normalization=False, threshold=25):
        self.normalization = normalization
        self.threshold = threshold

    def standardize(self, x):
        nn = x.shape[0] #number of data points
        mm = x.shape[1] #number of features
        me = [] # grand means
        mmax = [] # maximum value
        mmin = [] # minimum value
        ranges = [] # ranges

        for j in range(mm): # for each feature
            z = x[:, j]
            me.append(np.mean(z))
            mmax.append(np.max(z))
            mmin.append(np.min(z))
            if not self.normalization:
                ranges.append(1)
            else:
                ranges.append(mmax[j] - mmin[j])
            if ranges[j] == 0:
                print("Variable num {} is contant!".format(j))
                ranges[j] = 1

        sy = np.divide((x - me), ranges)
        sY = np.array(sy)
        d = np.sum(sY * sY)   # total data scatter of normalized data
        if self.normalization:
            self.sY = sY
        else:
            self.sY = sY
        self.data = x
        self.ranges = ranges
        self.me = me
        
        return nn, me, ranges, d
        
    def fit_transform(self,x,max_clusters=None):
        nn, me, ranges, d = self.standardize(x)
        
        ancl = [] # data structure to keep everything together
        remains = list(range(nn)) # current index set of residual data after some anomalous clusters are extracted
        numberC = 0; # anomalous cluster counter
        while(len(remains) != 0):
            distance = dist(x, remains, ranges, me) # normalised distance vector from remains data points to reference 'me'
            ind = np.argmax(distance)
            index = remains[ind]
            centroid = x[index, :] # initial anomalous center reference point
            numberC = numberC + 1

            (cluster, centroid) = anomalousPattern(x, remains, ranges, centroid, me) # finding AP cluster

            censtand = np.divide((np.asarray(centroid) - me), np.asarray(ranges)) # standardised centroid   
            dD = np.sum(np.divide(censtand * censtand.T * len(cluster) * 100, d))   # cluster contribution, per cent 

            remains = np.setdiff1d(remains, cluster) 
            # update the data structure that keeps everything together
            ancl.append(cluster)   # set of data points in the cluster
            ancl.append(censtand)  # standardised centroid
            ancl.append(dD) # proportion of the data scatter

            if max_clusters is not None and numberC == max_clusters:
                break

        ancl = np.asarray(ancl, dtype=object)
        ancl = ancl.reshape((numberC, 3))

        ##aK = numberC
        b = 3
        ll = [] # list of clusters

        for ik in range(numberC):
            ll.append(len(ancl[ik, 0]))

        rl = [i for i in ll if i > self.threshold] # list of clusters with more than threshold elements
        cent = []
        clusters_confirmed = []
        points_remain = []
        
        if(len(rl) == 0):
            print('Too great a threhsold!!!')
        else:
            num_cents = 0
            for ik in range(numberC):
                cluster = ancl[ik,0]
                if(len(cluster) > self.threshold):
                    cent.append(ancl[ik, 1])
                    num_cents += 1
                    clusters_confirmed.append(cluster)
                else:
                    for i in cluster:
                        points_remain.append(i)

        self.cent = np.asarray(cent)

        remains = np.add(remains,1)
    
        return self.cent, self.sY, clusters_confirmed, remains
    
    def build_report(self, clusters, centroids_std):
        data_mean = self.me
        data_range = np.max(self.data, axis=0) - np.min(self.data, axis=0)
        data_range = self.ranges

        print("Features involved:")
        for i in range(self.data.shape[1]):
            print(f'Feature {i+1} Mean: {np.round(data_mean[i], 2)}')

        print()
        
        for i in range(len(clusters)):
            cluster_n = i+1
            cluster = clusters[cluster_n]
            print(f'Cluster {cluster_n} ({len(cluster)}):')
            print(cluster)

            centroid_std = centroids_std[i]
            
            if self.normalization:
                centroid = centroid_std*data_range + data_mean
            else:
                centroid = centroid_std+data_mean
            print(f'Cluser centroid (real): {[round(float(x),2) for x in centroid]}')

            print(f'Cluser centroid (stand): {[round(float(x),3) for x in centroid_std]}')

            centroid_perc = np.trunc((centroid * 100) / data_mean - 100)
            print(f'Centroid(% over/under grand mean): {centroid_perc}')

            print()