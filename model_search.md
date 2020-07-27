

```julia
using MLJ
```

```julia
models()
```

    Dict{Any,Any} with 9 entries:
      "MultivariateStats" => Any["ICA", "RidgeRegressor", "KernelPCA", "PCA"]
      "MLJ"               => Any["MLJ.Constant.DeterministicConstantRegressor", "ML…
      "DecisionTree"      => Any["DecisionTreeRegressor", "DecisionTreeClassifier"]
      "ScikitLearn"       => Any["SVMLRegressor", "SVMNuClassifier", "ElasticNet", …
      "LIBSVM"            => Any["EpsilonSVR", "LinearSVC", "NuSVR", "NuSVC", "SVC"…
      "Clustering"        => Any["KMeans", "KMedoids"]
      "GLM"               => Any["OLSRegressor", "GLMCountRegressor"]
      "NaiveBayes"        => Any["GaussianNBClassifier", "MultinomialNBClassifier"]
      "XGBoost"           => Any["XGBoostCount", "XGBoostRegressor", "XGBoostClassi…



```julia
task = supervised(inputs=Table(Continous),
                  target=AbstractVector{<:Finite},
                  is_probabilistic=true)
```


```julia
models(task)
```


    Dict{Any,Any} with 4 entries:
      "MLJ"          => Any["ConstantClassifier"]
      "DecisionTree" => Any["DecisionTreeClassifier"]
      "NaiveBayes"   => Any["GaussianNBClassifier"]
      "XGBoost"      => Any["XGBoostClassifier"]



