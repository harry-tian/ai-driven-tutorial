#Import necessary packages
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import STAP
#Must be activated

class ProtoclassExplainer():
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
    
    def explain(self, X, Z, labels, e):
        dxz = self.dist2(X, Z)
        prot = self.protoclass(X, labels, Z, dxz, e)
        prot_dict = {key:item for key,item in prot.items()}
        return prot_dict