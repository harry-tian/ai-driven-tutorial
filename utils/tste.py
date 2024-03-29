#!/usr/bin/env python2
"""TSTE: t-Distributed Stochastic Triplet Embedding

X = tste(triplets, no_dims=2, lambda=0, alpha=no_dims-1, use_log=True, verbose=True)

where 'triplets' is an integer array with shape (k, 3) containing the
known triplet relationships between points. Each point is refered to
by its index: consider a set of points and some (possibly nonmetric)
distance function d.

For each t in triplets' rows,
   assume d(point[t[0]], point[t[1]])
        < d(point[t[0]], point[t[2]])

The function implements t-distributed stochastic triplet embedding
(t-STE) based on the specified triplets, to construct an embedding
with no_dims dimensions. The parameter lambda specifies the amount of
L2- regularization (default = 0), whereas alpha sets the number of
degrees of freedom of the Student-t distribution (default = 1). The
variable use_log determines whether the sum of the log-probabilities
or the sum of the probabilities is maximized (default = true).

Note: This function directly learns the embedding.


Original MATLAB implementation: (C) Laurens van der Maaten, 2012, Delft University of Technology

Python and Theano port: (C) Michael Wilber, 2013, UC San Diego

"""



USING_THEANO = True

import numpy as np

try:
    import theano as t
    import theano.tensor as T
except ImportError:
    USING_THEANO=False
    import warnings
    warnings.warn("Please go install Theano for a ~2-3x speedup. Until then, I'm falling back to the slower pure Python implementation.", UserWarning)


def tste(triplets, no_dims=2, lamb=0, alpha=None, use_log=True,verbose=False, max_iter=100, save_each_iteration=False, initial_X=None,static_points=np.array([]), normalize_gradient=False, ignore_zeroindexed_error=True):
    """Learn the triplet embedding for the given triplets.

    Returns an array with shape (max(triplets)+1, no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.

    """
    if alpha is None:
        alpha = no_dims-1

    N = np.max(triplets) + 1
    assert -1 not in triplets

    # A warning to Matlab users:
    if not ignore_zeroindexed_error:
        assert 0 in triplets, "Triplets should be 0-indexed, not 1-indexed!"
    # Technically, this is allowed I guessss... if your triplets don't
    # refer to some points you need... Just don't say I didn't warn
    # you. Remove this assertion at your own peril!

    n_triplets = len(triplets)

    # Initialize some variables
    if initial_X is None:
        X = np.random.randn(N, no_dims) * 0.0001
    else:
        X = initial_X

    C = np.Inf
    tol = 1e-7              # convergence tolerance
    eta = 2.                # learning rate
    best_C = np.Inf         # best error obtained so far
    best_X = X              # best embedding found so far
    iteration_Xs = []       # for debugging ;) *shhhh*

    # Perform main iterations
    iter = 0; no_incr = 0;
    while iter < max_iter and no_incr < 5:
        old_C = C;
        # Calculate gradient descent and cost
        if USING_THEANO:
            if use_log:
                C,G = tste_grad_theano_log(X, N, no_dims, triplets, lamb, alpha)
            else:
                C,G = tste_grad_theano(X, N, no_dims, triplets, lamb, alpha)
        else:
            C,G = tste_grad_python(X, N, no_dims, triplets, lamb, alpha, use_log)

        if C < best_C:
            best_C = C
            best_X = X

        # Perform gradient update
        if save_each_iteration:
            iteration_Xs.append(X.copy())

        # (NEW:) Optionally normalize each point's gradient by the
        # number of times that the point appears in the triplets
        if normalize_gradient:
            prior = np.bincount(triplets.ravel(),
                                minlength=N)
            prior[prior==0] = 1
            # if prior[i]==0, then the point has no gradient anyway
            G = G / (prior/np.linalg.norm(prior))[:,np.newaxis]
        X = X - (float(eta) / n_triplets * N) * G
        if len(static_points):
            X[static_points] = initial_X[static_points]

        # Update learning rate
        if old_C > C + tol:
            no_incr = 0
            eta *= 1.01
        else:
            no_incr = no_incr + 1
            eta *= 0.5

        # Print out progress
        iter += 1
        if verbose and iter%10 == 0:
            # These are Euclidean distances:
            sum_X = np.sum(X**2, axis=1)
            D = -2 * (X.dot(X.T)) + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
            # ^ D = squared Euclidean distance?
            no_viol = np.sum(D[triplets[:,0],triplets[:,1]] > D[triplets[:,0],triplets[:,2]]);
            print ("Iteration ",iter, ' error is ',C,', number of constraints: ', (float(no_viol) / n_triplets))

    if save_each_iteration:
        return iteration_Xs

    print(get_triplet_acc(best_X, triplets))

    return best_X

def tste_grad_python(X, N, no_dims, triplets, lamb, alpha, use_log):
    """Computes the cost function and corresponding gradient of t-STE. A
    purely python implementation."""

    triplets_A = triplets[:,0].copy()
    triplets_B = triplets[:,1].copy()
    triplets_C = triplets[:,2].copy()

    # Compute Student-t kernel
    sum_X = np.sum(X**2, axis=1)
    a = -2 * (X.dot(X.T))
    b = a + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
    # ^ something like the squared Euclidean distance?
    K = (1 + b / alpha) ** ((alpha+1)/-2)
    # ^ Student-T kernel

    # Compute value of cost function
    P = K[triplets_A,triplets_B] / (
        K[triplets_A,triplets_B] +
        K[triplets_A,triplets_C])
    if use_log:
        # Pp = P.copy()
        # Pp[P <= 0] = np.finfo(np.float64).tiny
        C = -np.sum(np.log(P)) + lamb * np.sum(X**2)
    else:
        C = -np.sum(P) + lamb * np.sum(X**2);
    # Compute gradient for each point
    dC = np.zeros((N, no_dims))
    for i in range(no_dims):
        # For i = each dimension to use
        const = (alpha+1) / alpha
        if use_log:
            A_to_B = (1 - P) * K[triplets_A,triplets_B] * (X[triplets_A, i] - X[triplets_B, i])
            B_to_C = (1 - P) * K[triplets_A,triplets_C] * (X[triplets_A, i] - X[triplets_C, i])
        else:
            A_to_B = P * (1-P) * K[triplets_A, triplets_B] * (X[triplets_A,i] - X[triplets_B,i])
            B_to_C = P * (1-P) * K[triplets_A, triplets_C] * (X[triplets_A,i] - X[triplets_C,i])
        this_dim = -const * np.array([A_to_B - B_to_C,
                                      -A_to_B,
                                      B_to_C]).T
        # Group up each point into our gradient
        # for n_triplet in range(N):
        #     dC[n_triplet, i] = np.sum(this_dim*(triplets == n_triplet))

        # Doing it this way avoids python for loops.
        dC[:, i] = np.bincount(triplets.ravel(),
                                   minlength=N,
                                   weights=this_dim.ravel())

    dC = -dC + 2*lamb*X
    return C, dC

# Theano-flavored tste_grad
if USING_THEANO:
    def make_theano_evaluator(use_log):
        """This returns a function(!) that calculates the gradient and cost. Heh."""
        X = T.dmatrix('X')
        triplets = T.imatrix('triplets')
        alpha = T.dscalar('alpha')
        lamb = T.dscalar('lambda')
        no_dims = T.iscalar('no_dims')
        N = T.iscalar('N')
        triplets_A = triplets[:,0]
        triplets_B = triplets[:,1]
        triplets_C = triplets[:,2]

        # Compute Student-t kernel. Look familiar?
        sum_X = T.sum(X**2, axis=1)
        a = -2 * (X.dot(X.T))
        b = a + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
        K = (1 + b / alpha) ** ((alpha+1)/-2)

        # Compute value of cost function
        P = K[triplets_A,triplets_B] / (
            K[triplets_A,triplets_B] +
            K[triplets_A,triplets_C])
        if use_log:
            C = -T.sum(T.log(P)) + lamb * T.sum(X**2)
        else:
            C = -T.sum(P)        + lamb * T.sum(X**2)

        # Compute gradient, for each dimension
        const = (alpha+1) / alpha

        dim = T.iscalar('dim')
        def each_dim(dim):
            if use_log:
                A_to_B =   (1 - P) * K[triplets_A,triplets_B] * (X[triplets_A][:,dim] - X[triplets_B][:,dim])
                B_to_C =   (1 - P) * K[triplets_A,triplets_C] * (X[triplets_A][:,dim] - X[triplets_C][:,dim])
            else:
                A_to_B = P*(1 - P) * K[triplets_A,triplets_B] * (X[triplets_A][:,dim] - X[triplets_B][:,dim])
                B_to_C = P*(1 - P) * K[triplets_A,triplets_C] * (X[triplets_A][:,dim] - X[triplets_C][:,dim])
            this_dim = (-const * T.stack(A_to_B - B_to_C, -A_to_B, B_to_C)).T

            dC = T.extra_ops.bincount(triplets.ravel(),
                                      weights=this_dim.ravel(),
                                      # minlength=N
                                      )
            return -dC + 2*lamb*X[:,dim]

        # loop across all dimensions... theano loops are weird, yes...
        all_dims = (t.scan(each_dim,
                           # non_sequences=N,
                            sequences=T.arange(no_dims))
                   )[0].T

        return t.function([X,N,no_dims,triplets,lamb,alpha],
                          [C, all_dims],
                          on_unused_input='ignore')

    tste_grad_theano_log = make_theano_evaluator(use_log=True)
    tste_grad_theano = make_theano_evaluator(use_log=False)

def euc_dist(x, y): return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def get_triplet_acc(embeds, triplets, dist_f=euc_dist):
    ''' Return triplet accuracy given ground-truth triplets.''' 
    align = []
    for triplet in triplets:
        a, p, n = triplet
        ap = dist_f(embeds[a], embeds[p]) 
        an = dist_f(embeds[a], embeds[n])
        align.append(ap < an)
    acc = np.mean(align)
    return acc