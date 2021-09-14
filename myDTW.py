'''
Author: Siyun WANG
Version: 2
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class myDTW():
    def __init__(self):
        self.switch = False
    
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
            self.switch = len(a) > len(b)
            diff = abs(len(a)-len(b))
            shorter = a if not self.switch else b
            longer = a if self.switch else b
            self.a = shorter
            self.b = longer
            self.distMat = np.ones((len(shorter)+1,len(longer)+1))*np.inf
            self.distMat[0,0] = 0
            for i in range(1,len(shorter)+1):
                for j in range(max(1, i-window+1),min(len(longer)+1,i+diff+window)):
                    val1 = abs(shorter[i-1]-longer[j-1])
                    val2 = min(self.distMat[i-1, j-1], self.distMat[i-1,j], self.distMat[i,j-1])
                    self.distMat[i,j] = val1 + val2
        self.distance = self.distMat[i, j]
        
    def compute_matrix_deprecated(self, a, b, window, correlationSize):
        '''deprecated'''
        if window is None:
            self.a = a
            self.b = b
            self.distMat = np.ones((len(a)+1,len(b)+1))*np.inf
            self.distMat[0,0] = 0
            for i in range(1,len(a)+1):
                for j in range(1,len(b)+1):
                    val1 = abs(a[i-1]-b[j-1])
                    val2 = min(self.distMat[i-1, j-1], self.distMat[i-1,j], self.distMat[i,j-1])
                    # new: add correlation information into the distance matrix
                    if correlationSize > 0:
                        if (i>3) & (j>3): #(at least 3 points to get non-trivial correlation)
                            validLen = min(correlationSize, i-1, j-1)
                            val3 = np.corrcoef(a[i-validLen-1:i], b[j-validLen-1:j])[0,1]
                        else:
                            val3 = 0
                    else:
                        val3 = 0
                    self.distMat[i,j] = val1 + val2 - val3
        else:
            self.switch = len(a) > len(b)
            diff = abs(len(a)-len(b))
            shorter = a if not self.switch else b
            longer = a if self.switch else b
            self.a = shorter
            self.b = longer
            self.distMat = np.ones((len(shorter)+1,len(longer)+1))*np.inf
            self.distMat[0,0] = 0
            for i in range(1,len(shorter)+1):
                for j in range(max(1, i-window+1),min(len(longer)+1,i+diff+window)):
                    val1 = abs(shorter[i-1]-longer[j-1])
                    val2 = min(self.distMat[i-1, j-1], self.distMat[i-1,j], self.distMat[i,j-1])
                    if correlationSize > 0:
                        if (i>3) & (j>3): #(at least 3 points to get non-trivial correlation)
                            validLen = min(correlationSize, i-1, j-1)
                            val3 = np.corrcoef(shorter[i-validLen-1:i], longer[j-validLen-1:j])[0,1]
                        else:
                            val3 = 0
                    else:
                        val3 = 0
                    self.distMat[i,j] = val1 + val2 - val3
        self.distance = self.distMat[i, j]
        
    def back_tracking(self):
        # initialisations
        self.pathLength = 1
        i = self.distMat.shape[0] - 1
        j = self.distMat.shape[1] - 1
        
        axis0 = [i]
        axis1 = [j]
        while (i>1) | (j>1):
            i, j = self._helper_get_next_pivot(i, j)
            axis0.append(i)
            axis1.append(j)
            self.pathLength += 1
        
        self.pathMat = np.zeros_like(self.distMat)
        self.pathMat[axis0, axis1] = 1
        self.pathMat = self.pathMat[1:,1:]
        
        if self.switch:
            self.axis_a = np.array(axis1)
            self.axis_b = np.array(axis0)
        else:
            self.axis_a = np.array(axis0)
            self.axis_b = np.array(axis1)
        
    def _helper_get_next_pivot(self,i,j):
        tmpMat = self.distMat[i-1:i+1, j-1:j+1].copy()
        tmpMat[-1,-1] = np.inf
        minIndex = np.argmin(tmpMat)
        minIndices = (minIndex//2, minIndex%2)
        return i-(1-minIndices[0]), j-(1-minIndices[1])
    
    def plot_optimal_path(self):
        fig = plt.figure(figsize=(12,12))
        G = plt.GridSpec(ncols=4,nrows=4,figure=fig)
        
        axA = fig.add_subplot(G[1:,0])
        axA.plot(self.a,np.arange(len(self.a)))
        axA.set(xticks=[])
        axB = fig.add_subplot(G[0,1:])
        axB.plot(self.b)
        axB.set(yticks=[])
        axPath = fig.add_subplot(G[1:,1:],sharey=axA,sharex=axB)
        axPath.imshow(self.pathMat,'gray')
        axPath.set(xticks=[], yticks=[])
        plt.show()
    
    def compute(self, a, b, window=None, returnLength=False):
        self.compute_matrix(a, b, window)
        self.back_tracking()
        if returnLength:
            return self.distance/self.pathLength, self.pathLength
        else:
            return self.distance #/self.pathLength

    def match_period(self, start, end, a2b=True):
        '''
        matches a period from a curve to another

        start, end: indices of the period to be matched
        a2b: bool, default to True. If true, match a period from curve a's timeline to b's and vice versa if false.

        returns the tuple of the matched period (start, end) of the other curve
        '''
        start = max(0, start)
        if a2b:
            if not self.switch:
                end = min(end, len(self.a)-1)
                return np.where(self.pathMat[start,:]==1)[0][0], np.where(self.pathMat[end,:]==1)[0][0]
            else:
                end = min(end, len(self.b)-1)
                return np.where(self.pathMat[:,start]==1)[0][0], np.where(self.pathMat[:,end]==1)[0][0]
        else:
            if not self.switch:
                end = min(end, len(self.b)-1)
                return np.where(self.pathMat[:,start]==1)[0][0], np.where(self.pathMat[:,end]==1)[0][0]
            else:
                end = min(end, len(self.a)-1)
                return np.where(self.pathMat[start,:]==1)[0][0], np.where(self.pathMat[end,:]==1)[0][0]
