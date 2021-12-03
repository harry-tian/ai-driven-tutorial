python version of Bien and Tibshirani's (2012) Protoclass algorithm

### prerequisites:
install `R` and `rpy2` (a python package)

### usage
parameters:
- `X` is the dataset to select prototypes from
- `Z` is prototype candidates, usually X=Z
- `Y` is the labels of X

To specify number of prototypes `m`, use `protoclass_m` which returns list of prototype indices with respect to `X`

`proto_idx = protoclass_m(X, Z, Y, m)`

To specify list of m `m_range`,  use `protoclass_mrange` which returns dictionary with `m` as keys and lists of prototype indices as values

`proto_dict = protoclass_mrange(X, Z, Y, m_range)`

To specify eps, use `ProtoclassExplainer().explain()` which returns the prototype indices and a `R` dictionary

`explainer = ProtoclassExplainer()`

`idx, prot = explainer.explain(X, Z, Y, eps)`
