from __future__ import print_function

import numpy as np
import xport
from sklearn.preprocessing import OneHotEncoder
from cvxopt.solvers import qp
from cvxopt import matrix, spmatrix
from numpy import array, ndarray
from scipy.spatial.distance import cdist
from qpsolvers import solve_qp
from sklearn.metrics.pairwise import rbf_kernel
import sys

def runOptimiser(K, u, b, l, preOptw, initialValue, maxWeight=10000):
    """
    Args:
        K (double 2d array): Similarity/distance matrix
        u (double array): Mean similarity of each prototype
        preOptw (double): Weight vector
        initialValue (double): Initialize run
        maxWeight (double): Upper bound on weight
    Returns:
        Prototypes, weights and objective values
    """
    d = u.shape[0]
    lb = np.zeros((d, 1))
    ub = maxWeight * np.ones((d, 1))
    x0 = np.append( preOptw, initialValue/K[d-1, d-1] )

    G = np.vstack((np.identity(d), -1*np.identity(d)))
    h = np.vstack((ub, -1*lb))

    #     Solve a QP defined as follows:
    #         minimize
    #             (1-l)/2 * x.T * P * x + (q.T - l * r.T) * x
    #         subject to
    #             G * x <= h
    #             A * x == b
    sol = solve_qp(K, (l * b - u), G, h, A=None, b=None, solver='cvxopt', initvals=x0)

    # compute objective function value
    x = sol.reshape(sol.shape[0], 1)
    P = K
    q = - u.reshape(u.shape[0], 1)
    r = - b.reshape(b.shape[0], 1)
    # obj_value = (1-l)/2 * np.matmul(np.matmul(x.T, P), x) + np.matmul(q.T, x) - l * np.matmul(r.T, x)
    obj_value = 1/2 * np.matmul(np.matmul(x.T, P), x) + np.matmul(q.T, x) - l * np.matmul(r.T, x)
    return(sol, obj_value[0,0])


def pdash_e(X, Y, B, m, kernelType='other', sigma=2, l=1):
    """
    Main prototype selection function.
    Args:
        X (double 2d array): Dataset to explain
        Y (double 2d array): Dataset to select
        B (double 2d array): Dataset to exclude
        m (double): Number of prototypes
        kernelType (str): Gaussian, linear or other
        sigma (double): Gaussian kernel width
    Returns:
        Current optimum, the prototypes and objective values throughout selection
    """
    numY = Y.shape[0]
    numX = X.shape[0]
    numB = B.shape[0]
    allY = np.array(range(numY))

    # Store the mean inner products with X
    if kernelType == 'Gaussian':
        meanInnerProductX = np.zeros((numY, 1))
        meanInnerProductB = np.zeros((numY, 1))
        for i in range(numY):
            Y1 = Y[i, :]
            Y1 = Y1.reshape(Y1.shape[0], 1).T
            distX = cdist(X, Y1)
            distB = cdist(B, Y1)
            meanInnerProductX[i] = np.sum( np.exp(np.square(distX)/(-2.0 * sigma**2)) ) / numX
            meanInnerProductB[i] = np.sum( np.exp(np.square(distB)/(-2.0 * sigma**2)) ) / numB
    else:
        M = np.dot(Y, np.transpose(X))
        meanInnerProductX = np.sum(M, axis=1) / M.shape[1]
        M = np.dot(Y, np.transpose(B))
        meanInnerProductB = np.sum(M, axis=1) / M.shape[1]

    # move to features x observation format to be consistent with the earlier code version
    X = X.T
    B = B.T
    Y = Y.T

    # Intialization
    S = np.zeros(m, dtype=int)
    setValues = np.zeros(m)
    sizeS = 0
    currSetValue = 0.0
    currOptw = np.array([])
    currK = np.array([])
    curru = np.array([])
    currb = np.array([])
    runningInnerProduct = np.zeros((m, numY))

    while sizeS < m:

        remainingElements = np.setdiff1d(allY, S[0:sizeS])

        newCurrSetValue = currSetValue
        maxGradient = 0

        for count in range(remainingElements.shape[0]):

            i = remainingElements[count]
            newZ = Y[:, i]

            if sizeS == 0:

                if kernelType == 'Gaussian':
                    K = 1
                else:
                    K = np.dot(newZ, newZ)

                u = meanInnerProductX[i]
                b = meanInnerProductB[i]
                w = np.max(u / K, 0)
                # incrementSetValue =  (l-1) / 2 * K * (w ** 2) + (u * w) - l * (b * w)
                incrementSetValue = 1 / 2 * K * (w ** 2) + (u * w) - l * (b * w)

                if (incrementSetValue > newCurrSetValue) or (count == 0):
                    # Bookeeping
                    newCurrSetValue = incrementSetValue
                    desiredElement = i
                    newCurrOptw = w
                    currK = K

            else:
                recentlyAdded = Y[:, S[sizeS - 1]]

                if kernelType == 'Gaussian':
                    distnewZ = np.linalg.norm(recentlyAdded-newZ)
                    runningInnerProduct[sizeS - 1, i] = np.exp( np.square(distnewZ)/(-2.0 * sigma**2 ) )
                else:
                    runningInnerProduct[sizeS - 1, i] = np.dot(recentlyAdded, newZ)

                innerProduct = runningInnerProduct[0:sizeS, i]
                if innerProduct.shape[0] > 1:
                    innerProduct = innerProduct.reshape((innerProduct.shape[0], 1))

                # gradientVal = meanInnerProductX[i] - l * meanInnerProductB[i] + (l - 1) * np.dot(currOptw, innerProduct)
                gradientVal = meanInnerProductX[i] - l * meanInnerProductB[i] - np.dot(currOptw, innerProduct)

                if (gradientVal > maxGradient) or (count == 0):
                    maxGradient = gradientVal
                    desiredElement = i
                    newinnerProduct = innerProduct[:]

        S[sizeS] = desiredElement

        curru = np.append(curru, meanInnerProductX[desiredElement])
        currb = np.append(currb, meanInnerProductB[desiredElement])

        if sizeS > 0:

            if kernelType == 'Gaussian':
                selfNorm = array([1.0])
            else:
                addedZ = Y[:, desiredElement]
                selfNorm = array( [np.dot(addedZ, addedZ)] )

            K1 = np.hstack((currK, newinnerProduct))

            if newinnerProduct.shape[0] > 1:
                selfNorm = selfNorm.reshape((1,1))
            K2 = np.vstack( (K1, np.hstack((newinnerProduct.T, selfNorm))) )

            currK = K2
            if maxGradient <= 0:
                #newCurrOptw = np.vstack((currOptw[:], np.array([0])))
                newCurrOptw = np.append(currOptw, [0], axis=0)
                newCurrSetValue = currSetValue
            else:
                [newCurrOptw, value] = runOptimiser(currK, curru, currb, l, currOptw, maxGradient)
                newCurrSetValue = -value

        currOptw = newCurrOptw
        if type(currOptw) != np.ndarray:
            currOptw = np.array([currOptw])

        currSetValue = newCurrSetValue

        setValues[sizeS] = currSetValue
        sizeS = sizeS + 1

    return(currOptw, S, setValues)


def proto_g(X, candidate_indices, m, is_K_sparse=False, gamma=0.125, K=None):

    if K is None:
        K = rbf_kernel(X, gamma=gamma)

    if len(candidate_indices) != np.shape(K)[0]:
        K = K[:,candidate_indices][candidate_indices,:]

    n = len(candidate_indices)

    # colsum = np.array(K.sum(0)).ravel() # same as rowsum
    if is_K_sparse:
        colsum = 2*np.array(K.sum(0)).ravel() / n
    else:
        colsum = 2*np.sum(K, axis=0) / n

    selected = np.array([], dtype=int)
    value = np.array([])
    for i in range(m):
        maxx = -sys.float_info.max
        argmax = -1
        candidates = np.setdiff1d(range(n), selected)

        s1array = colsum[candidates]
        if len(selected) > 0:
            temp = K[selected, :][:, candidates]
            if is_K_sparse:
                # s2array = temp.sum(0) *2
                s2array = temp.sum(0) * 2 + K.diagonal()[candidates]

            else:
                s2array = np.sum(temp, axis=0) *2 + np.diagonal(K)[candidates]

            s2array = s2array/(len(selected) + 1)

            s1array = s1array - s2array

        else:
            if is_K_sparse:
                s1array = s1array - (np.abs(K.diagonal()[candidates]))
            else:
                s1array = s1array - (np.abs(np.diagonal(K)[candidates]))

        argmax = candidates[np.argmax(s1array)]
        # print("max %f" %np.max(s1array))

        selected = np.append(selected, argmax)
        # value = np.append(value,maxx)
        KK = K[selected, :][:, selected]
        if is_K_sparse:
            KK = KK.todense()

        inverse_of_prev_selected = np.linalg.inv(KK)  # shortcut

    return candidate_indices[selected]