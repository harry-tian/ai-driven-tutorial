from __future__ import print_function

import numpy as np
# import xport
from sklearn.preprocessing import OneHotEncoder
from cvxopt.solvers import qp
from cvxopt import matrix, spmatrix
from numpy import array, ndarray
from scipy.spatial.distance import cdist
from qpsolvers import solve_qp
from sklearn.metrics.pairwise import rbf_kernel
import sys
import utils

def NN_align(index, S, f_m, f_h, is_triplet=False):
    S = np.concatenate([S,[index]])
    S_f_m = f_m[S]
    S_f_h = f_h[S]

    score = 0
    for i in range(len(S)):
        if f_h.shape[0] == f_h.shape[1:]:
            nn_S_f_h = utils.get_1nn_lpips(i, S_f_h)
        else:
            nn_S_f_h = utils.get_1nn(i, S_f_h)
        if utils.get_1nn(i, S_f_m) == nn_S_f_h:
            score += 1

    return score

def NN_label_constraint(index, f_m, f_m_labels, f_h, f_h_labels, is_triplet=True):
    NN_f_m = utils.get_1nn(index, f_m)
    if is_triplet:
        NN_f_h = utils.get_nn_triplets(index, f_h)
    else:
        NN_f_h = utils.get_1nn(index, f_h)

    if (f_m_labels[NN_f_m] == f_h_labels[NN_f_h]).any():
        return True
    else:
        return False


def pdash(X, Y, m, f_h, f_h_scale=0.00001, is_triplet=False, kernelType="other", sigma=2):
    """
    Main prototype selection function.
    Args:
        X (double 2d array): Dataset to select prototypes from
        Y (double 2d array): Dataset to explain
        m (double): Number of prototypes
        kernelType (str): Gaussian, linear or other
        sigma (double): Gaussian kernel width
    Returns:
        Current optimum, the prototypes and objective values throughout selection
    """
    numY = Y.shape[0]
    numX = X.shape[0]
    allYIndices = np.array(range(numY))

    # Store the mean inner products with X
    if kernelType == 'Gaussian':
        meanInnerProductX = np.zeros((numY, 1))
        for i in range(numY):
            Y1 = Y[i, :]
            Y1 = Y1.reshape(Y1.shape[0], 1).T
            distX = cdist(X, Y1)
            meanInnerProductX[i] = np.sum( np.exp(np.square(distX)/(-2.0 * sigma**2)) ) / numX
    else:
        M = np.dot(Y, np.transpose(X))
        meanInnerProductX = np.sum(M, axis=1) / M.shape[1]

    # move to features x observation format to be consistent with the earlier code version
    X = X.T
    Y = Y.T

    # Intialization
    S = np.zeros(m, dtype=int)
    setValues = np.zeros(m)
    sizeS = 0
    currSetValue = 0.0
    currOptw = np.array([])
    currK = np.array([])
    curru = np.array([])
    runningInnerProduct = np.zeros((m, numY))

    while sizeS < m:

        remainingIndices = np.setdiff1d(allYIndices, S[0:sizeS])

        newCurrSetValue = currSetValue
        maxGradient = 0
        maxScore = 0

        for count in range(remainingIndices.shape[0]):

            i = remainingIndices[count]
            newZ = Y[:, i]

            if sizeS == 0:

                if kernelType == 'Gaussian':
                    K = 1
                else:
                    K = np.dot(newZ, newZ)

                u = meanInnerProductX[i]
                w = np.max(u / K, 0)
                incrementSetValue = -0.5 * K * (w ** 2) + (u * w)

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

                gradientVal = meanInnerProductX[i] - np.dot(currOptw, innerProduct)

                ############ add triplet constraint here ###########
                align_score = NN_align(i, S, Y.T, f_h, is_triplet=is_triplet) * f_h_scale
                score = gradientVal + align_score
                # print(gradientVal, align_score)

                if (score > maxScore) or (count == 0):
                    maxScore = score
                    maxGradient = gradientVal
                    desiredElement = i
                    newinnerProduct = innerProduct[:]

        S[sizeS] = desiredElement

        curru = np.append(curru, meanInnerProductX[desiredElement])

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
                [newCurrOptw, value] = runOptimiser_pdash(currK, curru, currOptw, maxGradient)
                newCurrSetValue = -value

        currOptw = newCurrOptw
        if type(currOptw) != np.ndarray:
            currOptw = np.array([currOptw])

        currSetValue = newCurrSetValue

        setValues[sizeS] = currSetValue
        sizeS = sizeS + 1

    return(currOptw, S, setValues)

def runOptimiser_pdash(K, u, preOptw, initialValue, maxWeight=10000):
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
    #             (1/2) * x.T * P * x + q.T * x
    #         subject to
    #             G * x <= h
    #             A * x == b
    sol = solve_qp(K, -u, G, h, A=None, b=None, solver='cvxopt', initvals=x0)

    # compute objective function value
    x = sol.reshape(sol.shape[0], 1)
    P = K
    q = - u.reshape(u.shape[0], 1)
    obj_value = 1/2 * np.matmul(np.matmul(x.T, P), x) + np.matmul(q.T, x)
    return(sol, obj_value[0,0])