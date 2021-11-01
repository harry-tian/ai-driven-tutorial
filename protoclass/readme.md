# example:
X is the dataset to select prototypes from
Y is the list of labels of X (assert len(X)==len(Y))
eps is the size of covering balls in the protoclass algorithm: larger eps leads to less number of prototypes

`from protoclass import ProtoclassExplainer`
`explainer = ProtoclassExplainer()`
`Z=X`
`prot = explainer.explain(X, Z, Y, eps=250000)`