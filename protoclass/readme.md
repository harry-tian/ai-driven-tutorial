python version of Bien and Tibshirani's (2012) Protoclass algorithm

### prerequistes:
install `R` and `rpy2` (a python package)

### parameters:

`X` is the dataset to select prototypes from

`Z` is the list of prototype candidates, usually X=Z

`labels` is the labels of X 

`eps` is the size of covering balls: larger `eps` leads to less number of prototypes

### returns:

`idx`: index (in X) of selected prototypes

`prot`: an instance of the "protoclass" class as defined in `protoclass.r`, contains additional information about the prototypes

### example code

`from protoclass import ProtoclassExplainer`

`explainer = ProtoclassExplainer()`

`idx, prot = explainer.explain(X, X, labels, eps=250000)`
