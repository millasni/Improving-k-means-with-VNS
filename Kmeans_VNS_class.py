#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random as ra

class K_Means():
    
    # random_state sets seed
    def __init__(self, k=2, tol=0.00001, max_iter=300, max_iter_VNS = 5, random_state=12): # Added max_iter_VNS
        self.k = k # number of clusters
        self.tol = tol          #tolerance used to stop the kmeans
        self.max_iter = max_iter #max # iter. to make if tolerance not reached
        self.max_iter_VNS = max_iter_VNS #max iter. of VNS
        self.random_state = random_state # set seed for reprodusibility
        
    def cluster(self,data,initialize=True,VNS_centroids={}):
        
        """
        Description : function that creates the clusters
        Input : data = Array of samples to cluster
        Output : dictionnary of clusters
        """
        if initialize==True:
            self.centroids = {} # creating empty dictionary of centroids
            # for each cluster - select randomly an observation as a centroid
            for i in range(self.k): 
                
                # Setting random seed for reproducibility
                ra.seed(self.random_state) # Sometimes reproducible, sometimes not...
                random_idx = np.random.permutation(data.shape[0])# data.shape[0]=N
                # randomly select data points as centroids according to top-K 
                self.centroids[i] = data[random_idx[i]] 
        else: 
            self.centroids = VNS_centroids
            
            
            # For each iteration we create a dictionary of classes - 
            # we are going to class data with each cluster
        for i in range(self.max_iter):
            self.clustering = {} # creating empty dictionary of clusters

            for j in range(self.k):
                self.clustering[j] = [] # initiating empty array for cluster j
                # in a dictionary of clusters
                
            # initiate an empty list to store min distances to the centroids
            # need this to calculate withing-cluster sum of squared errors 
            for sample in data:
                # this for loop of centroid calculation can be parallelized
                # calculating distances between each sample and each centroid
                sample_distances = [np.linalg.norm(sample - self.centroids[centroid]) for centroid in self.centroids]
                # taking centroid with the minimum distance
                cluster = sample_distances.index(min(sample_distances))
                # adding the sample to the dictionary of clusters 
                self.clustering[cluster].append(sample)
                # saving min distance from a sample to its centroid
                
            # calculating withing-cluster sum of squared errors    
        
            #Update des nouveaux centroïdes et on conserve les anciens centroïdes pour l'optimisation en tolérance
            previous_centroids = dict(self.centroids)
            for cluster in self.clustering:
                self.centroids[cluster] = np.average(self.clustering[cluster], axis=0)
            
            #Calculation of update and stopping criteria
            stop_dist = np.sum([np.linalg.norm(previous_centroids[centroid] - self.centroids[centroid]) for centroid in self.centroids])
            
            if i > 0:
                if stop_dist < self.tol:
                    break
        
        # calculating withing-cluster sum of squared errors
        within_cl_distances = []
        for cluster in range(self.k):            
            # calculating distances between each sample and each centroid
            within_cl_distances.append(sum([np.linalg.norm(sample - self.centroids[cluster]) for sample in self.clustering[cluster]]))

        # calculating withing-cluster sum of squared errors    
        self.SSE = sum(within_cl_distances) 

        return self.SSE, self.centroids, self.clustering 
    
    # this is just for storing best solution while VNS is working. 
    # not to be run by itself after VNS is done: the last tried solution will be stored,
    # which might not be the best one.
    def best_sol(self):
        self.SSE_best = self.SSE.copy()
        self.centroids_best = self.centroids.copy()
        self.clusters_best = self.clustering.copy()
        print("Best SSE = ",self.SSE_best)
        return self.SSE_best, self.centroids_best, self.clusters_best

    def shaking(self, k=1): 
        kmax = self.k 
        centroids_old = self.centroids_best.copy() 
        centroids_new = self.centroids_best.copy() 
        l = list(range(kmax)) # create list with cluster numbers
        ra.shuffle(l) # shuffle randomly cluster list
        d = list(range(len(data))) # create list with data indexes
        ra.shuffle(d) # shuffle randomly data indexes
        r_centr = l[:k] # taking top-k shuffled cluster numbers - these centroids will be shaken
        r_obs = d[:k] # taking top-k shuffled data indexes - these become new centroids
        for j in range(k):
#             #  assigning random observation(s) to centroid(s) we're shaking
            centroids_new[r_centr[j]] = data[r_obs[j]] # replacing chosen cenroids with random data points
        return centroids_old, centroids_new 
    
    # VNS combines shaking, local search (cluster()) and neighbourhood change.
    def VNS(self, data):
        self.best_sol()
        counter=1
        kmax = self.k # maximum number of centroids to be shaken = K
        k=1
        # VNS main loop
        while counter <= self.max_iter_VNS:
            print("---------> counter =",counter,",",k," centroids replaced")
            # shaking centroids
            shake_res = self.shaking(k)
            centr_new = shake_res[1]
            # local search with new centroids
            VNS_sol = self.cluster(data, initialize=False, VNS_centroids=centr_new)
            SSE_VNS = VNS_sol[0]
            if SSE_VNS < self.SSE_best:
                print("Improved SSE = ",SSE_VNS)
                self.best_sol()
                self.VNS_best = VNS_sol
                k=1
                counter=1
            # neighbourhood change if no improvement at lower neighbourhood
            else:
                print("Worse SSE = ",SSE_VNS)
                k+=1
                if k > kmax:
                    k=1
                    counter+=1
        print("End of VNS search, best SSE = ",self.SSE_best)
        return self.VNS_best

