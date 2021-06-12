'''
Author: Siyun WANG
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class myDTW():
    def __init__(self):
        pass
    
    def compute_matrix(self, a, b, window):
        
        if window is None:
            self.a = a
            self.b = b
            self.distMat = np.ones((len(a)+1,len(b)+1))*np.inf
            self.distMat[0,0] = 0
            for i in range(1,len(a)+1):
                for j in range(1,len(b)+1):
                    val1 = abs(a[i-1]-b[j-1])
                    val2 = min(self.distMat[i-1, j-1], self.distMat[i-1,j], self.distMat[i,j-1])
                    self.distMat[i,j] = val1 + val2
        else:
            diff = abs(len(a)-len(b))
            shorter = a if len(a) < len(b) else b
            longer = a if len(a) >= len(b) else b
            self.a = shorter
            self.b = longer
            self.distMat = np.ones((len(shorter)+1,len(longer)+1))*np.inf
            self.distMat[0,0] = 0
            for i in range(1,len(shorter)+1):
                for j in range(max(1, i-window+1),min(len(longer)+1,i+diff+window)): #%diff
                    val1 = abs(shorter[i-1]-longer[j-1])
                    val2 = min(self.distMat[i-1, j-1], self.distMat[i-1,j], self.distMat[i,j-1])
                    self.distMat[i,j] = val1 + val2
        self.distance = self.distMat[i, j]
        
    def back_tracking(self):
        # initialisations
        self.pathLength = 1
        i = self.distMat.shape[0] - 1
        j = self.distMat.shape[1] - 1
        
        self.axis0 = [i]
        self.axis1 = [j]
        while (i>1) | (j>1):
            i, j = self._helper_get_next_pivot(i, j)
            #self.distance += self.distMat[i,j]
            self.axis0.append(i)
            self.axis1.append(j)
            self.pathLength += 1
             
        self.axis0 = np.array(self.axis0)
        self.axis1 = np.array(self.axis1)
        
    def _helper_get_next_pivot(self,i,j):
        tmpMat = self.distMat[i-1:i+1, j-1:j+1].copy()
        tmpMat[-1,-1] = np.inf
        minIndex = np.argmin(tmpMat)
        minIndices = (minIndex//2, minIndex%2)
        return i-(1-minIndices[0]), j-(1-minIndices[1])
    
    def plot_optimal_path(self):
        matcp = self.distMat.copy()
        matcp[self.axis0, self.axis1] = 1e6
        fig = plt.figure(figsize=(12,12))
        G = plt.GridSpec(ncols=4,nrows=4,figure=fig)
        
        axA = fig.add_subplot(G[1:,0])
        axA.plot(self.a,np.arange(len(self.a)))
        axA.set(xticks=[])
        axB = fig.add_subplot(G[0,1:])
        axB.plot(self.b)
        axB.set(yticks=[])
        axPath = fig.add_subplot(G[1:,1:],sharey=axA,sharex=axB)
        axPath.imshow(matcp[1:,1:],'gray')
        axPath.set(xticks=[], yticks=[])
        plt.show()
    
    def compute(self, a, b, window=None, returnLength=False):
        self.compute_matrix(a, b, window)
        self.back_tracking()
        if returnLength:
            return self.distance/self.pathLength, self.pathLength
        else:
            return self.distance #/self.pathLength
