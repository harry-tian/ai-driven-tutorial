import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP
import numpy as np

class ProtoclassExplainer():
    """
    ProtoclassExplainer uses rpy2 to convert Bien and Tibshirani's (2012) Protoclass algorithm in R to python

    This is implemented by mimicking the DIExplainer class in aix360
    """
    def __init__(self, path_r_file="protoclass.r"):
        pandas2ri.activate()
        with open(path_r_file, 'r') as f:
            file = f.read()

        protoclass = STAP(file,"protoclass")
        dist2 = STAP(file,"dist2")
        predict_protoclass = STAP(file,"predict.protoclass")
        greedy = STAP(file,"greedy")
        analyzeSolution = STAP(file,"analyzeSolution")
        # plot_protoclass = STAP(file,"plot.protoclass")
        # print_protoclass = STAP(file,"print.protoclass")

        self.protoclass = protoclass.protoclass
        self.dist2 = dist2.dist2
        self.predict_protoclass = predict_protoclass.predict_protoclass
        self.greedy = greedy.greedy
        self.analyzeSolution = analyzeSolution.analyzeSolution
        # self.plot_protoclass = plot_protoclass.plot_protoclass
        # self.print_protoclass = print_protoclass.print_protoclass
    
    def explain(self, X, Z, Y, eps, lamda=None):
        '''
        Return index of selected prototypes and an instance of the "protoclass" class 

        Args: 
            X (double 2d array): Dataset you want to explain.
            Z (double 2d array): Dataset to select prototypes from.
            Y (double 2d array): Labels of X 
            eps (float): size of covering ball

        Returns:
            idx: index (in X) of selected prototypes
            prot: an instance of the "protoclass" class as defined in protoclass.r
        '''
        dxz = self.dist2(X, Z)
        if not lamda:
            lamda = 1/len(Y)
        prot = self.protoclass(X, Y, Z, dxz, eps, lamda)
        prot_dict = {key:item for key,item in prot.items()}
        idx = self.proto_idx(prot_dict)
        return idx, prot_dict

    def proto_idx(self, prot):
        alpha = [sum(x) for x in prot["alpha"]]
        idx = []
        for i, a in enumerate(alpha):
            if a > 0:
                idx.append(i)
            
        return idx

def protoclass_mrange(X, Z, Y, m_range, lamda=None, eps_step=None, find_min_eps=False, debug=False):
    '''
    Takes a m_range list where each m is the number of prototypes to be returned
    Returns index of selected prototypes in list and dictionary format

    Args: 
        X (double 2d array): Dataset you want to explain.
        Z (double 2d array): Dataset to select prototypes from.
        Y (double 2d array): Labels of X 
        m_range (list): list of int m
        lamda: cost of adding a prototype, default is 1/len(Y), can tune this to be around 1-5
        eps_step: 
        find_min_eps: 
        debug

    Returns:
        m_dict: dictionary: key is m and value is prototypes
    '''
    if not lamda:
        lamda = 1/(len(Y))

    a_dim = X[:,0]
    b_dim = X[:,1]
    a_range = max(a_dim) - min(a_dim)
    b_range = max(b_dim) - min(b_dim)
    min_range = min(a_range, b_range)
    max_range = max(a_range, b_range)
    m_dict = {m:[] for m in m_range}
    if 0 in m_dict.keys():
        m_dict[0] = ["?"]

    if not eps_step:
        #### tune this part
        if min_range < 5 and min_range > 2:
            ## for 2 <= range <= 5, set eps_step = 0.1
            eps_step = 0.1
        elif min_range < 2:
            eps_step = 0.05
        else:
            eps_step = 0.2
            
    while True:
        eps_range = np.arange(0, max_range, eps_step)[1:]
        if find_min_eps:
            eps_range = np.flip(eps_range) 
            

        for eps in eps_range:
            protoclass = ProtoclassExplainer()
            train_pclass_idx, train_prot = protoclass.explain(X, Z, Y, eps, lamda=lamda)
            m = len(train_pclass_idx)
            
            if m in m_range:
                m_dict[m] = train_pclass_idx

        if debug:
            print(m_dict)

        # if np.array(list(m_dict.values()),dtype=object).all(): ????????????
        if not ([] in m_dict.values()):
            break
        else:
            eps_step /= 2

#     prototype_idss = m_dict.values()
    return m_dict

def protoclass_m(X, Z, Y, m, lamda=None, eps_step=None, find_min_eps=False, debug=False):
    m_range = [m]
    proto_dict = protoclass_mrange(X, Z, Y, m_range, lamda=None, eps_step=None, find_min_eps=False, debug=False)
    return proto_dict[m]