#Import necessary packages
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP
#Must be activated

class ProtoclassExplainer():
    """
    ProtoclassExplainer uses rpy2 to convert Bien and Tibshirani's (2012) Protoclass algorithm in R to python

    This is implemented by mimicking the DIExplainer class in aix360
    """
    def __init__(self):
        pandas2ri.activate()
        with open('protoclass.r', 'r') as f:
            file = f.read()
        protoclass = STAP(file,"protoclass")
        print_protoclass = STAP(file,"print.protoclass")
        dist2 = STAP(file,"dist2")
        predict_protoclass = STAP(file,"predict.protoclass")
        greedy = STAP(file,"greedy")
        analyzeSolution = STAP(file,"analyzeSolution")
        plot_protoclass = STAP(file,"plot.protoclass")

        self.protoclass = protoclass.protoclass
        self.dist2 = dist2.dist2
        self.predict_protoclass = predict_protoclass.predict_protoclass
        self.greedy = greedy.greedy
        self.analyzeSolution = analyzeSolution.analyzeSolution
        self.plot_protoclass = plot_protoclass.plot_protoclass
        self.print_protoclass = print_protoclass.print_protoclass
    
    def explain(self, X, Z, labels, eps, lamda=None):
        '''
        Return index of selected prototypes and an instance of the "protoclass" class 

        Args: 
            X (double 2d array): Dataset you want to explain.
            Z (double 2d array): Dataset to select prototypes from.
            labels (double 2d array): Labels of X 
            eps (float): size of covering ball

        Returns:
            idx: index (in X) of selected prototypes
            prot: an instance of the "protoclass" class as defined in protoclass.r
        '''
        dxz = self.dist2(X, Z)
        if not lamda:
            lamda = 1/len(labels)
        prot = self.protoclass(X, labels, Z, dxz, eps, lamda)
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

def protoclass(X, Z, labels, m_range, lamda=None, eps_step=None, find_min_eps=False):
    '''
    Takes a m_range list where each m is the number of prototypes to be returned
    Returns index of selected prototypes in list and dictionary format

    Args: 
        X (double 2d array): Dataset you want to explain.
        Z (double 2d array): Dataset to select prototypes from.
        labels (double 2d array): Labels of X 
        m_range (list): list of int m
        lamda: cost of adding a prototype, default is 1/len(labels), can tune this to be around 1-5
        eps_step: search-related parameter
        find_min_eps: 

    Returns:
        prototype_idss: list of prototypes in index format
        m_dict: dictionary: key is m and value is prototypes
    '''
    if not lamda:
        lamda = 1/(len(labels))

    a_dim = X[:,0]
    b_dim = X[:,1]
    a_range = max(a_dim) - min(a_dim)
    b_range = max(b_dim) - min(b_dim)
    min_range = min(a_range, b_range)
    max_range = max(a_range, b_range)
    m_dict = {m:[] for m in m_range}

    if not eps_step:
        #### tune this part
        if min_range < 5 and min_range > 2:
            ## for 2 <= range <= 5, set eps_step = 0.1
            eps_step = 0.1
        elif min_range < 2:
            eps_step = 0.05
        else:
            eps_step = 0.2
    # print(eps_step)
    while True:
        eps_range = np.arange(0, max_range, eps_step)[1:]
        if find_min_eps:
            eps_range = np.flip(eps_range) 
        # print(eps_range)
        # print(m_dict)
        for eps in eps_range:
            # print(eps)
            protoclass = ProtoclassExplainer()
            train_pclass_idx, train_prot = protoclass.explain(X, Z, labels, eps, lamda=lamda)
            m = len(train_pclass_idx)
            
            if m in m_range:
                m_dict[m] = train_pclass_idx

        if np.array(list(m_dict.values()),dtype=object).all():
            break
        else:
            eps_step /= 2

    prototype_idss = [m_dict[m] for m in m_range]
    return prototype_idss, m_dict
