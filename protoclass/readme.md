python version of Bien and Tibshirani's (2012) Protoclass algorithm

### prerequisites:
install `R` and `rpy2` (a python package)

### usage

To specify number of prototypes:

`proto_dict = protoclass(X, Z, Y, m_range)`

where `X` is the dataset to select prototypes from, `Z` is prototype candidates, usually X=Z, `Y` is the labels of X, and `m_range` is the list of number of prototypes.

To specify eps:

`explainer = ProtoclassExplainer()`

`idx, prot = explainer.explain(X, Z, Y, eps)`
