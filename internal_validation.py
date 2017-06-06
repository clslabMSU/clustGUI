# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 16:15:37 2017

"""
import numpy as np

import scipy as sy

class internalIndex:
    def  __init__(self, num_k):
        self.num_k = num_k
        self.class_iter = range(1, num_k +1)
    
    def euclidean_centroid(self, data,label, label_num = False):
        if label_num == False:
            num_attr = data.shape[1]
            centroid = np.zeros([1, num_attr])
            for attr_id in range(num_attr):
                sum_attr_id = 0
                for i in range(len(data)):
                    sum_attr_id += data[i][attr_id]
                centroid[0][attr_id] = sum_attr_id/len(data)
            return centroid[0]
        else:
            count = 0
            for l in label:
                if l == label_num:
                    count +=1
            length = count
            num_attr = data.shape[1]
            centroid = np.zeros([1, num_attr])
            for attr_id in range(num_attr):
                sum_attr_id = 0
                for i in range(len(data)):
                    if label[i] == label_num:
                        sum_attr_id += data[i][attr_id]
                centroid[0][attr_id] = sum_attr_id/length
            return centroid[0]
    def centroid_list(self, data,label):
        c_list = []
        for index_i in self.class_iter:
            c_list.append(self.euclidean_centroid(data,label,index_i))
        return c_list
        
    def element_of_clustert (self, data, label, cluster_i):
        eoc = []
        for l in range(len(data)):
            if label[l] == cluster_i:
                eoc.append(data[l])
        return np.asarray(eoc)
    
    def distance_from_cluster (self, data, label,  cluster_i, centroid_i):
        eoc = self.element_of_clustert(data, label, cluster_i)
        centroid_i = self.euclidean_centroid(data, label, centroid_i)
        distance = 0
        for i in eoc:
            distance += sy.spatial.distance.euclidean(centroid_i, i)
        return distance
    
    def distance_from_cluster_sqr (self, data, label,  cluster_i, centroid_i):
        eoc = self.element_of_clustert(data, label, cluster_i)
        centroid_i = self.euclidean_centroid(data, label, centroid_i)
        distance = 0
        for i in eoc:
            distance += sy.spatial.distance.sqeuclidean(centroid_i, i)
        return distance
    
#    def cluster_stdev(self, data, label, i = False):
#        if i == 'all':
#            result = 0
#            for c in self.class_iter:
#                result +=cluster_stdev(data, label,c)
#            return (np.sqrt(result)) / num_k
#        if i !=False:
#            data = self.element_of_clustert(data, label, i)
#        var_vec = np.var(data, 0 )
#        var_vec_t = np.transpose(var_vec)
#        return np.sqrt(np.dot(var_vec,var_vec_t))
    
    def dbi (self, data, label):
        db = 0
        for index_i in self.class_iter:
            c_i = self.euclidean_centroid(data, label, index_i)
            max_rij=0
            d_i_avg = self.distance_from_cluster(data, label, index_i, index_i)/ len(self.element_of_clustert(data, label, index_i))
            for index_j in self.class_iter:
                if index_i == index_j:
                    continue
                else:
                    c_j = self.euclidean_centroid(data,label, index_j)
                    d_j_avg = self.distance_from_cluster(data, label, index_j, index_j)/ len(self.element_of_clustert(data, label, index_j))
                    d_i_j = sy.spatial.distance.euclidean(c_j, c_i)
                    candidate = (d_i_avg + d_j_avg) /d_i_j
                    if candidate > max_rij:
                        max_rij = candidate
            db+=max_rij
        return db/(len(np.unique(label)))
    
    def xie_benie(self, data, label):
        total_distance = 0
        for index_i in self.class_iter:
            total_distance += self.distance_from_cluster_sqr(data, label, index_i, index_i)
        c_list = self.centroid_list(data,label)
        min_cij = sy.spatial.distance.pdist(c_list,'sqeuclidean').min()
        xb = total_distance / (len(data) * min_cij)
        return xb
    
    def dunn(self, data, label):
        min_ij_candidate = float('Infinity')
        max_ck = float('-Infinity')
        for index_k in self.class_iter:
            eoc_k = self.element_of_clustert(data, label, index_k)
            if len(eoc_k) == 1 :
                candidate = 0
            else:
                candidate = sy.spatial.distance.pdist(eoc_k,'euclidean').max()
            if candidate > max_ck:
                max_ck =candidate
        for index_i in self.class_iter:
            eoc_i = self.element_of_clustert(data, label, index_i)
            for index_j in self.class_iter:
                if index_i == index_j:
                    continue
                eoc_j = self.element_of_clustert(data, label, index_j)
                min_j = float('Infinity')
                for e_c_i in eoc_i:
                    for e_c_j in eoc_j:
                        candidate = sy.spatial.distance.euclidean(e_c_i, e_c_j )
                        if candidate <  min_j:
                            min_j = candidate
                        else:
                            pass
                min_j = min_j /max_ck
                if  min_j < min_ij_candidate:
                    min_ij_candidate = min_j
        return min_ij_candidate
                    
    def CH(self, data, label):
        data_centroid = self.euclidean_centroid(data, label)
        cent_distsqr = 0
        ecent_distsqr= 0 
        for index_i in self.class_iter:
            n_element_i  = len(self.element_of_clustert(data, label, index_i))
            ci_centroid = self.euclidean_centroid(data, label, index_i)
            sqr_dist = sy.spatial.distance.sqeuclidean(data_centroid, ci_centroid)
            cent_distsqr = cent_distsqr + sqr_dist * n_element_i
            ecent_distsqr += self.distance_from_cluster_sqr(data, label, index_i, index_i)
        return (cent_distsqr / (self.num_k - 1)) / (ecent_distsqr / (len(data) - self.num_k ))
