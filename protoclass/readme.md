python version of Bien and Tibshirani's (2012) Protoclass algorithm

### parameters:

`X` is the dataset to select prototypes from

`labels` is the list of labels of X (assert len(X)==len(Y))

`Z` is the list of prototype candidates, usually X=Z

`eps` is the size of covering balls: larger `eps` leads to less number of prototypes

### example code

`from protoclass import ProtoclassExplainer`

`explainer = ProtoclassExplainer()`

`prot = explainer.explain(X, X, labels, eps=250000)`
