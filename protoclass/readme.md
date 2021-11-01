
`X` is the dataset to select prototypes from

`labels` is the list of labels of X (assert len(X)==len(Y))

`Z` is the list of prototype candidates, usually X=Z

`eps` is the size of covering balls in the protoclass algorithm: larger `eps` leads to less number of prototypes

## example code

`from protoclass import ProtoclassExplainer`

`explainer = ProtoclassExplainer()`

`Z=X`

`prot = explainer.explain(X, Z, labels, eps=250000)`
