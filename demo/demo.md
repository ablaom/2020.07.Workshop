

```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
```

    [32m[1m  Updating[22m[39m registry at `~/.julia/registries/General`
    [32m[1m  Updating[22m[39m git-repo `https://github.com/JuliaRegistries/General.git`
    [?25l[2K[?25h


```julia
using MLJ
```

    ┌ Info: Loading model metadata
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/MLJ.jl:114


## Getting some data:


```julia
using RDatasets
iris = dataset("datasets", "iris"); # a DataFrame
scrambled = shuffle(1:size(iris, 1))
X = iris[scrambled, 1:4];
y = iris[scrambled, 5];

first(X, 4)
```




<table class="data-frame"><thead><tr><th></th><th>SepalLength</th><th>SepalWidth</th><th>PetalLength</th><th>PetalWidth</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>4 rows × 4 columns</p><tr><th>1</th><td>7.2</td><td>3.2</td><td>6.0</td><td>1.8</td></tr><tr><th>2</th><td>5.0</td><td>3.5</td><td>1.3</td><td>0.3</td></tr><tr><th>3</th><td>5.0</td><td>3.5</td><td>1.6</td><td>0.6</td></tr><tr><th>4</th><td>5.7</td><td>2.9</td><td>4.2</td><td>1.3</td></tr></tbody></table>




```julia
y[1:5]
```




    5-element CategoricalArray{String,1,UInt8}:
     "virginica" 
     "setosa"    
     "setosa"    
     "versicolor"
     "setosa"    



## Basic fit and predict:


```julia
@load SVC()
classifier_ = SVC()
classifier = machine(classifier_, X, y)
fit!(classifier)
ŷ = predict(classifier, X) # or some Xnew
```

    import MLJModels ✔
    import LIBSVM ✔
    import MLJModels.LIBSVM_.SVC ✔


    ┌ Info: Training [34mMachine{SVC} @ 1…12[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:140


    *
    optimization finished, #iter = 33
    nu = 0.038907
    obj = -1.945147, rho = -0.167869
    nSV = 10, nBSV = 0
    *
    optimization finished, #iter = 48
    nu = 0.293514
    obj = -21.377494, rho = -0.144367
    nSV = 33, nBSV = 26
    *
    optimization finished, #iter = 35
    nu = 0.046521
    obj = -2.403410, rho = 0.039522
    nSV = 10, nBSV = 2
    Total nSV = 44





    150-element Array{CategoricalString{UInt8},1}:
     "virginica" 
     "setosa"    
     "setosa"    
     "versicolor"
     "setosa"    
     "virginica" 
     "versicolor"
     "versicolor"
     "virginica" 
     "versicolor"
     "virginica" 
     "virginica" 
     "virginica" 
     ⋮           
     "setosa"    
     "versicolor"
     "versicolor"
     "setosa"    
     "virginica" 
     "virginica" 
     "versicolor"
     "setosa"    
     "versicolor"
     "versicolor"
     "versicolor"
     "versicolor"



## Evaluating the model:


```julia
evaluate!(classifier,
          resampling=Holdout(fraction_train=0.8),
          measure=misclassification_rate)
# ## Adding dimension reduction:
@load PCA
dim_reducer_ = PCA()
dim_reducer = machine(dim_reducer_, X)
fit!(dim_reducer)
Xsmall = transform(dim_reducer, X);

first(Xsmall, 3)
```

    ┌ Info: Evaluating using a holdout set. 
    │ fraction_train=0.8 
    │ shuffle=false 
    │ measure=MLJ.misclassification_rate 
    │ operation=StatsBase.predict 
    │ Resampling from all rows. 
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/resampling.jl:100


    import MLJModels ✔
    import MultivariateStats ✔
    import MLJModels.MultivariateStats_.PCA ✔


    ┌ Info: Training [34mMachine{PCA} @ 1…98[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:140





<table class="data-frame"><thead><tr><th></th><th>x1</th><th>x2</th><th>x3</th></tr><tr><th></th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>3 rows × 3 columns</p><tr><th>1</th><td>-2.61409</td><td>0.560901</td><td>0.205535</td></tr><tr><th>2</th><td>2.7701</td><td>0.263528</td><td>-0.0772477</td></tr><tr><th>3</th><td>2.40561</td><td>0.188871</td><td>-0.263868</td></tr></tbody></table>




```julia
classifier = machine(classifier_, Xsmall, y)
fit!(classifier)
ŷ = predict(classifier, Xsmall)
```

    *
    optimization finished, #iter = 23
    nu = 0.038664
    obj = -1.933164, rho = -0.165650
    nSV = 8, nBSV = 0
    *
    optimization finished, #iter = 38
    nu = 0.293883
    obj = -21.597810, rho = -0.082448
    nSV = 34, nBSV = 26
    *
    optimization finished, #iter = 30
    nu = 0.045664
    obj = -2.380751, rho = 0.053250
    nSV = 9, nBSV = 2
    Total nSV = 45


    ┌ Info: Training [34mMachine{SVC} @ 1…52[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:140





    150-element Array{CategoricalString{UInt8},1}:
     "virginica" 
     "setosa"    
     "setosa"    
     "versicolor"
     "setosa"    
     "virginica" 
     "versicolor"
     "versicolor"
     "virginica" 
     "versicolor"
     "virginica" 
     "virginica" 
     "virginica" 
     ⋮           
     "setosa"    
     "versicolor"
     "versicolor"
     "setosa"    
     "virginica" 
     "virginica" 
     "versicolor"
     "setosa"    
     "versicolor"
     "versicolor"
     "versicolor"
     "versicolor"



## Building a composite model:

### Method 1: Compact syntax (but not generalizable):

(not implemented at time of talk)


```julia
# composite_ = @pipeline dim_reducer_ classifier_

# composite = machine(composite_, X, y)
# evaluate!(composite, measure=misclassification_rate)
```

### Method 2: Re-interpret unstreamlined code:


```julia
Xraw = X;
yraw = y;

X = source(Xraw)
y = source(yraw)

dim_reducer = machine(dim_reducer_, X)
Xsmall = transform(dim_reducer, X)

classifier = machine(classifier_, Xsmall, y)
ŷ = predict(classifier, Xsmall)
```




    [34mNode @ 1…92[39m = predict([0m[1m1…90[22m, transform([0m[1m1…02[22m, [34m5…24[39m))




```julia
fit!(ŷ)
```

    ┌ Info: Training [34mNodalMachine{PCA} @ 1…02[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:140
    ┌ Info: Training [34mNodalMachine{SVC} @ 1…90[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:140


    *
    optimization finished, #iter = 23
    nu = 0.038664
    obj = -1.933164, rho = -0.165650
    nSV = 8, nBSV = 0
    *
    optimization finished, #iter = 38
    nu = 0.293883
    obj = -21.597810, rho = -0.082448
    nSV = 34, nBSV = 26
    *
    optimization finished, #iter = 30
    nu = 0.045664
    obj = -2.380751, rho = 0.053250
    nSV = 9, nBSV = 2
    Total nSV = 45





    [34mNode @ 1…92[39m = predict([0m[1m1…90[22m, transform([0m[1m1…02[22m, [34m5…24[39m))




```julia
ŷ(rows=3:4)
```




    2-element Array{CategoricalString{UInt8},1}:
     "setosa"    
     "versicolor"




```julia
dim_reducer_.ncomp = 1  # maximum output dimension
fit!(ŷ)
```

    ┌ Info: Updating [34mNodalMachine{PCA} @ 1…02[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:152
    ┌ Info: Training [34mNodalMachine{SVC} @ 1…90[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:140


    *
    optimization finished, #iter = 13
    nu = 0.030533
    obj = -1.526884, rho = -0.270704
    nSV = 4, nBSV = 1
    *
    optimization finished, #iter = 20
    nu = 0.355841
    obj = -30.258034, rho = 0.019778
    nSV = 36, nBSV = 34
    *
    optimization finished, #iter = 8
    nu = 0.048815
    obj = -2.645552, rho = 0.204566
    nSV = 7, nBSV = 4
    Total nSV = 44





    [34mNode @ 1…92[39m = predict([0m[1m1…90[22m, transform([0m[1m1…02[22m, [34m5…24[39m))




```julia
ŷ(rows=3:4)
```




    2-element Array{CategoricalString{UInt8},1}:
     "setosa"    
     "versicolor"



 Changing classifier hyperparameter does not retrigger retraining of
 upstream dimension reducer:


```julia
classifier_.gamma = 0.1
fit!(ŷ)
```

    *
    optimization finished, #iter = 13
    nu = 0.033696
    obj = -1.838789, rho = -0.128178
    nSV = 5, nBSV = 2
    *
    optimization finished, #iter = 24
    nu = 0.429648
    obj = -35.588638, rho = -0.040530
    nSV = 44, nBSV = 42
    *
    optimization finished, #iter = 5
    nu = 0.080000
    obj = -4.676483, rho = -0.106043
    nSV = 8, nBSV = 8
    Total nSV = 53


    ┌ Info: Not retraining [34mNodalMachine{PCA} @ 1…02[39m.
    │  It appears up-to-date. Use force=true to force retraining.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:146
    ┌ Info: Updating [34mNodalMachine{SVC} @ 1…90[39m.
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/machines.jl:152





    [34mNode @ 1…92[39m = predict([0m[1m1…90[22m, transform([0m[1m1…02[22m, [34m5…24[39m))




```julia
ŷ(rows=3:4)
```




    2-element Array{CategoricalString{UInt8},1}:
     "setosa"    
     "versicolor"



Predicting on new data (`Xraw` in `source(Xraw)` is substituted for `Xnew`):


```julia
Xnew = (SepalLength = [4.0, 5.2],
        SepalWidth = [3.2, 3.0],
        PetalLength = [1.2, 1.5],
        PetalWidth = [0.1, 0.4],)
ŷ(Xnew)
```




    2-element Array{CategoricalString{UInt8},1}:
     "setosa"
     "setosa"



#### Exporting network as stand-alone reusable model:


```julia
composite_ = @from_network Composite(pca=dim_reducer_, svc=classifier_) <= (X, y, ŷ)
params(composite_)
```




    (pca = (ncomp = 1,
            method = :auto,
            pratio = 0.99,
            mean = nothing,),
     svc = (kernel = RadialBasis::KERNEL = 2,
            gamma = 0.1,
            weights = nothing,
            cost = 1.0,
            degree = 3,
            coef0 = 0.0,
            tolerance = 0.001,
            shrinking = true,
            probability = false,),)




```julia
composite = machine(composite_, Xraw, yraw)
evaluate!(composite, measure=misclassification_rate)
```

    ┌ Info: Evaluating using cross-validation. 
    │ nfolds=6. 
    │ shuffle=false 
    │ measure=MLJ.misclassification_rate 
    │ operation=StatsBase.predict 
    │ Resampling from all rows. 
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/resampling.jl:151
    [33mCross-validating: 100%[=========================] Time: 0:00:02[39m





    6-element Array{Float64,1}:
     0.08
     0.08
     0.0 
     0.12
     0.08
     0.04



## Evaluating a "self-tuning" random forest (nested resampling):


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



### Evaluating a single tree:


```julia
@load DecisionTreeRegressor # load code

tree_ = DecisionTreeRegressor(n_subfeatures=3)
tree = machine(tree_, task)
evaluate!(tree,
          resampling=Holdout(fraction_train=0.7),
          measure=[rms, mav])
```

    import MLJModels ✔
    import DecisionTree ✔
    import MLJModels.DecisionTree_.DecisionTreeRegressor ✔


    ┌ Info: Evaluating using a holdout set. 
    │ fraction_train=0.7 
    │ shuffle=false 
    │ measure=Function[rms, mav] 
    │ operation=StatsBase.predict 
    │ Resampling from all rows. 
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/resampling.jl:100





    (MLJ.rms = 8.795939100833767,
     MLJ.mav = 5.785953164160401,)



### Use ensembling wrapper to create a random forest:


```julia
forest_ = EnsembleModel(atom=tree_, n=10)
```




    MLJ.DeterministicEnsembleModel(atom = [34mDecisionTreeRegressor @ 7…75[39m,
                                   weights = Float64[],
                                   bagging_fraction = 0.8,
                                   rng = MersenneTwister(UInt32[0x08804db9, 0xfc38831f, 0xd5683001, 0x444075ec]),
                                   n = 10,
                                   parallel = true,
                                   out_of_bag_measure = Any[],)[34m @ 9…74[39m



### Wrapping in a tuning strategy creates a "self_tuning" random forest:


```julia
r1 = range(forest_, :bagging_fraction, lower=0.4, upper=1.0);
r2 = range(forest_, :(atom.n_subfeatures), lower=1, upper=12)

self_tuning_forest_ = TunedModel(model=forest_,
                          tuning=Grid(),
                          resampling=CV(),
                          ranges=[r1,r2],
                          measure=rms)
```




    MLJ.DeterministicTunedModel(model = [34mDeterministicEnsembleModel{DecisionTreeRegressor} @ 9…74[39m,
                                tuning = [34mGrid @ 2…87[39m,
                                resampling = [34mCV @ 1…01[39m,
                                measure = MLJ.rms,
                                operation = StatsBase.predict,
                                ranges = MLJ.NumericRange{T,Symbol} where T[[34mNumericRange @ 1…81[39m, [34mNumericRange @ 1…80[39m],
                                minimize = true,
                                full_report = true,
                                train_best = true,)[34m @ 6…25[39m



### Evaluate the self_tuning_forest (nested resampling):


```julia
self_tuning_forest = machine(self_tuning_forest_, task)

evaluate!(self_tuning_forest,
          resampling=CV(),
          measure=[rms,rmslp1])
```

    ┌ Info: Evaluating using cross-validation. 
    │ nfolds=6. 
    │ shuffle=false 
    │ measure=Function[rms, rmslp1] 
    │ operation=StatsBase.predict 
    │ Resampling from all rows. 
    └ @ MLJ /Users/anthony/.julia/packages/MLJ/tod7z/src/resampling.jl:151
    [33mCross-validating: 100%[=========================] Time: 0:00:18[39m





    (MLJ.rms = [2.91827, 3.40544, 4.60971, 4.54709, 8.12081, 3.79819],
     MLJ.rmslp1 = [0.148546, 0.119118, 0.148812, 0.134863, 0.345141, 0.221093],)


