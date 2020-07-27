```julia
using RDatasets
iris = dataset("datasets", "iris"); # a DataFrame
first(iris, 3)
```

<table class="data-frame"><thead><tr><th></th><th>SepalLength</th><th>SepalWidth</th><th>PetalLength</th><th>PetalWidth</th><th>Species</th></tr></thead><tbody><p>3 rows × 5 columns</p><tr><th>1</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>setosa</td></tr><tr><th>2</th><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>setosa</td></tr><tr><th>3</th><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td><td>setosa</td></tr></tbody></table>


```julia
task = supervised(data=iris,
                  target=:Species,
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

