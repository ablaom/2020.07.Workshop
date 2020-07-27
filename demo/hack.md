```julia
dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
fit!(dim_reducer)
Xsmall = transform(dim_reducer, X);
```

# cut

```julia
classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
fit!(classifier)
ŷ = predict(classifier, Xsmall)
```

# cut

```julia
dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
fit!(dim_reducer)
Xsmall = transform(dim_reducer, X);

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
fit!(classifier)
ŷ = predict(classifier, Xsmall)
```

# cut

```julia
composite_ = @pipeline dim_reducer_ classifier_ 
```

# cut

```julia
X = source(X)
y = source(y)

dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
fit!(dim_reducer)
Xsmall = transform(dim_reducer, X);

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
fit!(classifier)
ŷ = predict(classifier, Xsmall)
```

# cut

```julia
X = source(X)
y = source(y)

dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
Xsmall = transform(dim_reducer, X);

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
ŷ = predict(classifier, Xsmall)
```

# cut

```julia
X = source(X)
y = source(y)

dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
Xsmall = transform(dim_reducer, X);

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
ŷ = predict(classifier, Xsmall)

fit!(ŷ)
```

# cut

```julia
X = source(X)
y = source(y)

dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
Xsmall = transform(dim_reducer, X)

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
ŷ = predict(classifier, Xsmall)

fit!(ŷ)

ŷ(rows=3:4)
```

    2-element Array{CategoricalString{UInt8},1}:
     "versicolor"
     "versicolor"

# cut

```julia
Xnew = (SepalLength = [4.0, 5.2],
        SepalWidth = [3.2, 3.0],
        PetalLength = [1.2, 1.5],
        PetalWidth = [0.1, 0.4],)
ŷ(Xnew)
```

    2-element Array{CategoricalString{UInt8},1}:
     "setosa"
     "setosa"

# cut

```julia
composite = @from_network Composite(pca=dim_reducer_, svc=classifier_) <= (X, y, ŷ)
```
# cut

```julia
composite_ = @from_network Composite(pca=dim_reducer_, svc=classifier_) <= (X, y, ŷ)

composite = machine(composite_, X2, y2)
fit!(composite)
predict(composite, Xnew)

```

# cut

```julia
task = load_boston()
models(task)
```

    Dict{Any,Any} with 6 entries:
      "MultivariateStats" => Any["RidgeRegressor"]
      "MLJ"               => Any["MLJ.Constant.DeterministicConstantRegressor", "ML…
      "DecisionTree"      => Any["DecisionTreeRegressor"]
      "ScikitLearn"       => Any["SVMLRegressor", "ElasticNet", "ElasticNetCV", "SV…
      "LIBSVM"            => Any["EpsilonSVR", "NuSVR"]
      "XGBoost"           => Any["XGBoostRegressor"]

# cut

```julia
@load DecisionTreeRegressor # load code

tree_ = DecisionTreeRegressor(n_subfeatures=3)
tree = machine(tree_, task)
evaluate!(tree,
          resampling=Holdout(fraction_train=0.7),
          measure=[rms, mav])
```

# cut

```julia
@load DecisionTreeRegressor # load code

tree_ = DecisionTreeRegressor(n_subfeatures=3)
tree = machine(tree_, task)
evaluate!(tree,
          resampling=Holdout(fraction_train=0.7),
          measure=[rms, mav])

    (MLJ.rms = 8.795939100833767,
     MLJ.mav = 5.785953164160401,)
```

# cut

```julia
forest_ = EnsembleModel(atom=tree_, n=10)
```

# cut

```julia
forest_ = EnsembleModel(atom=tree_, n=10)

```

# cut

```julia
forest_ = EnsembleModel(atom=tree_, n=10)

r1 = range(forest_, :bagging_fraction, lower=0.4, upper=1.0);
r2 = range(forest_, :(atom.n_subfeatures), lower=1, upper=12)

self_tuning_forest_ = TunedModel(model=forest_,
                          tuning=Grid(),
                          resampling=CV(),
                          ranges=[r1,r2],
                          measure=rms)
```

# cut

```julia
forest_ = EnsembleModel(atom=tree_, n=10)

r1 = range(forest_, :bagging_fraction, lower=0.4, upper=1.0);
r2 = range(forest_, :(atom.n_subfeatures), lower=1, upper=12)

self_tuning_forest_ = TunedModel(model=forest_,
                          tuning=Grid(),
                          resampling=CV(),
                          ranges=[r1,r2],
                          measure=rms)

self_tuning_forest = machine(self_tuning_forest_, task)

evaluate!(self_tuning_forest,
          resampling=CV(),
          measure=[rms,rmslp1])
```

    (MLJ.rms = [2.91827, 3.40544, 4.60971, 4.54709, 8.12081, 3.79819],
     MLJ.rmslp1 = [0.148546, 0.119118, 0.148812, 0.134863, 0.345141, 0.221093],)


