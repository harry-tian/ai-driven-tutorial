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
