
## A tour of MLJ

### Models, machines, basic training and testing

Let's load data and define train and test rows:


```julia
using MLJ
using DataFrames, Statistics

Xraw = rand(300,3)
y = exp(Xraw[:,1] - Xraw[:,2] - 2Xraw[:,3] + 0.1*rand(300))
X = DataFrame(Xraw)

train, test = partition(eachindex(y), 0.70); # 70:30 split
```

A *model* is a container for hyperparameters:


```julia
knn_model=KNNRegressor(K=10)
```




    # [0m[1mKNNRegressor{Float64} @ 1…43[22m: 
    target_type             =>   Float64
    K                       =>   10
    metric                  =>   euclidean (generic function with 1 method)
    kernel                  =>   reciprocal (generic function with 1 method)
    




Wrapping the model in data creates a *machine* which will store training outcomes (called *fit-results*):


```julia
ensemble_model = EnsembleModel(atom=knn_model, n=20)
```




    # [0m[1mDeterministicEnsembleModel @ 1…89[22m: 
    atom                    =>   [0m[1mKNNRegressor{Float64} @ 1…43[22m
    weights                 =>   0-element Array{Float64,1}
    bagging_fraction        =>   0.8
    rng_seed                =>   0
    n                       =>   20
    parallel                =>   true
    





```julia
params(ensemble_model)
```




    (atom = (target_type = Float64, K = 20, metric = MLJ.KNN.euclidean, kernel = MLJ.KNN.reciprocal), weights = Float64[], bagging_fraction = 0.8, rng_seed = 0, n = 20, parallel = true)



To define a tuning grid, we construct ranges for the two parameters and collate these ranges following the same pattern above (omitting parameters that don't change):


```julia
B_range = range(ensemble_model, :bagging_fraction, lower= 0.5, upper=1.0, scale = :linear)
K_range = range(knn_model, :K, lower=1, upper=100, scale=:log10)
nested_ranges = (atom = (K = K_range,), bagging_fraction = B_range)
```




    (atom = (K = [0m[1mNumericRange @ 1…42[22m,), bagging_fraction = [0m[1mNumericRange @ 4…99[22m)



Now we choose a tuning strategy:


```julia
tuning = Grid(resolution=12)
```




    # [0m[1mGrid @ 7…31[22m: 
    resolution              =>   12
    parallel                =>   true
    




And a resampling strategy:


```julia
resampling = Holdout(fraction_train=0.8)
```




    # [0m[1mHoldout @ 1…02[22m: 
    fraction_train          =>   0.8
    shuffle                 =>   false
    




And define a new model which wraps the these strategies around our ensemble model:


```julia
tuned_ensemble_model = TunedModel(model=ensemble_model, 
    tuning=tuning, resampling=resampling, nested_ranges=nested_ranges)
```




    # [0m[1mDeterministicTunedModel @ 1…18[22m: 
    model                   =>   [0m[1mDeterministicEnsembleModel @ 1…89[22m
    tuning                  =>   [0m[1mGrid @ 7…31[22m
    resampling              =>   [0m[1mHoldout @ 1…02[22m
    measure                 =>   nothing
    operation               =>   predict (generic function with 22 methods)
    nested_ranges           =>   (omitted NamedTuple{(:atom, :bagging_fraction),Tuple{NamedTuple{(:K,),Tuple{MLJ.NumericRange{Int64,Symbol}}},MLJ.NumericRange{Float64,Symbol}}})
    full_report             =>   true
    




Fitting the corresponding machine tunes the underlying model (in this case an ensemble) and retrains on all supplied data:


```julia
tuned_ensemble = machine(tuned_ensemble_model, X[train,:], y[train])
fit!(tuned_ensemble);
```

    ┌ Warning: No measure specified. Using measure=MLJ.rms. 
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:82
    ┌ Info: Training [0m[1mMachine{MLJ.DeterministicTunedMo…} @ 1…91[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93
    [33mSearching a 132-point grid for best model: 100%[=========================] Time: 0:00:07[39m
    ┌ Info: Training best model on all supplied data.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/tuning.jl:142


For each fitted machine, one may access the fitted parameters (as opposed to the hyperparameters stored in its model). In the current case the "fitted parameter" is the best ensemble model (trained on all available data):


```julia
fp = fitted_params(tuned_ensemble)
```




    (best_model = [0m[1mDeterministicEnsembleModel @ 2…01[22m,)




```julia
@show fp.best_model.bagging_fraction
@show fp.best_model.atom.K;
```

    (fp.best_model).bagging_fraction = 0.9545454545454546
    ((fp.best_model).atom).K = 8


The `report` method gives more detail on the tuning process:


```julia
r=report(tuned_ensemble)           # named tuple
zip(keys(r), values(r)) |> collect # for better viewing
```




    5-element Array{Tuple{Symbol,Any},1}:
     (:parameter_names, ["atom.K" "bagging_fraction"])                                                                                                                                                                             
     (:parameter_scales, Symbol[:log10 :linear])                                                                                                                                                                                   
     (:parameter_values, Any[1 0.5; 2 0.5; … ; 66 1.0; 100 1.0])                                                                                                                                                                   
     (:measurements, [0.0943033, 0.088809, 0.0821, 0.0876285, 0.106594, 0.139502, 0.19676, 0.225126, 0.271769, 0.326425  …  0.130382, 0.0969401, 0.100561, 0.0768932, 0.0946774, 0.116214, 0.150653, 0.192588, 0.235204, 0.285023])
     (:best_measurement, 0.07367105860190028)                                                                                                                                                                                      



Evaluating the tuned model:


```julia
yhat = predict(tuned_ensemble, X[test,:])
rms(yhat, y[test])
```




    0.08409339267106415



Or, using all the data, get cross-validation estimates, with cv-tuning on each fold complement (nested resampling):


```julia
tuned_ensemble = machine(tuned_ensemble_model, X, y)
evaluate!(tuned_ensemble, resampling=CV(nfolds=4), verbosity=2)
```

    [33mCross-validating:  20%[=====>                   ]  ETA: 0:00:00[39m┌ Info: Training [0m[1mMachine{MLJ.DeterministicTunedMo…} @ 1…48[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93
    [33mSearching a 132-point grid for best model: 100%[=========================] Time: 0:00:06[39m
    ┌ Info: Training best model on all supplied data.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/tuning.jl:142
    [33mCross-validating:  40%[==========>              ]  ETA: 0:00:09[39m┌ Info: Training [0m[1mMachine{MLJ.DeterministicTunedMo…} @ 1…48[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93
    [33mSearching a 132-point grid for best model: 100%[=========================] Time: 0:00:06[39m
    ┌ Info: Training best model on all supplied data.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/tuning.jl:142
    [33mCross-validating:  60%[===============>         ]  ETA: 0:00:08[39m┌ Info: Training [0m[1mMachine{MLJ.DeterministicTunedMo…} @ 1…48[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93
    [33mSearching a 132-point grid for best model: 100%[=========================] Time: 0:00:06[39m
    ┌ Info: Training best model on all supplied data.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/tuning.jl:142
    [33mCross-validating:  80%[====================>    ]  ETA: 0:00:05[39m┌ Info: Training [0m[1mMachine{MLJ.DeterministicTunedMo…} @ 1…48[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93
    [33mSearching a 132-point grid for best model: 100%[=========================] Time: 0:00:06[39m
    ┌ Info: Training best model on all supplied data.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/tuning.jl:142
    [33mCross-validating: 100%[=========================] Time: 0:00:25[39m





    4-element Array{Float64,1}:
     0.08096082109810439
     0.07800556218551255
     0.07327732650564481
     0.08309441293645088



### Learning networks

MLJ has a flexible interface for building networks from multiple machine learning elements, whose complexity extend beyond linear "pipelines", and with a minimal of added abstraction.

In MLJ, a *learning network* is a graph whose nodes apply an operation, such as `predict` or `transform`, using a fixed machine (requiring training) - or which, alternatively, applies a regular (untrained) mathematical operation to its input(s). In practice, a learning network works with *fixed* sources for its training/evaluation data, but can be built and tested in stages. By contrast, an *exported learning network* is a learning network exported as a stand-alone, re-usable `Model` object, to which all the MLJ `Model`  meta-algorthims can be applied (ensembling, systematic tuning, etc). 

As we shall see, exporting a learning network as a reusable model, is very easy. 

### Building a simple learning network

![](wrapped_ridge.png)

The diagram above depicts a learning network which standardises the input data, `X`, learns an optimal Box-Cox transformation for the target, `y`, predicts new targets using ridge regression, and then inverse-transforms those predictions (for later comparison with the original test data). The machines are labelled yellow. 

To implement the network, we begin by loading all data needed for training and evaluation into *source nodes*:


```julia
Xs = source(X)
ys = source(y)
```




    [0m[1mSource @ 1…99[22m



We label nodes according to their outputs in the diagram. Notice that the nodes `z` and `yhat` use the same machine `box` for different operations. 

To construct the `W` node we first need to define the machine `stand` that it will use to transform inputs. 


```julia
stand_model = Standardizer()
stand = machine(stand_model, Xs)
```




    [0m[1mNodalMachine @ 1…82[22m = machine([0m[1mStandardizer @ 1…00[22m, [0m[1m8…05[22m)



Because `Xs` is a node, instead of concrete data, we can call `transform` on the machine without first training it, and the result is the new node `W`, instead of concrete transformed data:


```julia
W = transform(stand, Xs)
```




    [0m[1mNode @ 1…46[22m = transform([0m[1m1…82[22m, [0m[1m8…05[22m)



To get actual transformed data we *call* the node appropriately, which will require we first train the node. Training a node, rather than a machine, triggers training of *all* necessary machines in the network.


```julia
fit!(W, rows=train)
W()          # transform all data
W(rows=test) # transform only test data
W(X[3:4,:])  # transform any data, new or old
```

    ┌ Info: Training [0m[1mNodalMachine{Standardizer} @ 1…82[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93





    (x1 = [0.376715, 0.561488], x2 = [-0.213696, -0.555213], x3 = [-0.787951, 1.2842])



If you like, you can think of `W` (and the other nodes we will define) as "dynamic data": `W` is *data*, in the sense that  it an be called ("indexed") on rows, but *dynamic*, in the sense the result depends on the outcome of training events. 

The other nodes of our network are defined similarly:


```julia
box_model = UnivariateBoxCoxTransformer()  # for making data look normally-distributed
box = machine(box_model, ys)
z = transform(box, ys)

ridge_model = RidgeRegressor(lambda=0.1)
ridge =machine(ridge_model, W, z)
zhat = predict(ridge, W)

yhat = inverse_transform(box, zhat)
```




    [0m[1mNode @ 1…00[22m = inverse_transform([0m[1m2…26[22m, predict([0m[1m5…34[22m, transform([0m[1m1…82[22m, [0m[1m8…05[22m)))



We are ready to train and evaluate the completed network. Notice that the standardizer, `stand`, is *not* retrained, as MLJ remembers that it was trained earlier:


```julia
fit!(yhat, rows=train)
rms(y[test], yhat(rows=test)) # evaluate
```

    ┌ Info: Not retraining [0m[1mNodalMachine{Standardizer} @ 1…82[22m. It is up-to-date.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/networks.jl:201
    ┌ Info: Training [0m[1mNodalMachine{UnivariateBoxCoxTransfor…} @ 2…26[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93
    ┌ Info: Training [0m[1mNodalMachine{RidgeRegressor{Float64}} @ 5…34[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:93





    0.020077532519835517




```julia
yhat(X[3:4,:])  # predict on new or old data
```




    2-element Array{Float64,1}:
     0.6976639470833352
     0.244671334466375 



We can change hyperparameters and retrain:


```julia
ridge_model.lambda = 0.01
fit!(yhat, rows=train) 
rms(y[test], yhat(rows=test))
```

    ┌ Info: Not retraining [0m[1mNodalMachine{UnivariateBoxCoxTransfor…} @ 2…26[22m. It is up-to-date.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/networks.jl:201
    ┌ Info: Not retraining [0m[1mNodalMachine{Standardizer} @ 1…82[22m. It is up-to-date.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/networks.jl:201
    ┌ Info: Updating [0m[1mNodalMachine{RidgeRegressor{Float64}} @ 5…34[22m.
    └ @ MLJ /Users/anthony/Dropbox/Julia7/MLJ/src/machines.jl:97





    0.02008343668851138



> **Notable feature.** The machine, `ridge::NodalMachine{RidgeRegressor}`, is retrained, because its underlying model has been mutated. However, since the outcome of this training has no effect on the training inputs of the machines `stand` and `box`, these transformers are left untouched. (During construction, each node and machine in a learning network determines and records all machines on which it depends.) This behaviour, which extends to exported learning networks, means we can tune our wrapped regressor without re-computing transformations each time the hyperparameter is changed. 

### Exporting a learning network as a composite model

To export a learning network:
- Define a new `mutable struct` model type.
- Wrap the learning network code in a model `fit` method.

All learning networks that make determinisic (or, probabilistic) predictions export as models of subtype `Deterministic{Node}` (respectively, `Probabilistic{Node}`):



```julia
mutable struct WrappedRidge <: Deterministic{Node}
    ridge_model
end
```

Now satisfied that our wrapped Ridge Regression learning network works, we simply cut and paste its defining code into a `fit` method: 


```julia
function MLJ.fit(model::WrappedRidge, X, y)
    Xs = source(X)
    ys = source(y)

    stand_model = Standardizer()
    stand = machine(stand_model, Xs)
    W = transform(stand, Xs)

    box_model = UnivariateBoxCoxTransformer()  # for making data look normally-distributed
    box = machine(box_model, ys)
    z = transform(box, ys)

    ridge_model = model.ridge_model ###
    ridge =machine(ridge_model, W, z)
    zhat = predict(ridge, W)

    yhat = inverse_transform(box, zhat)
    fit!(yhat, verbosity=0)
    
    return yhat
end
```

The line marked `###`, where the new exported model's hyperparameter `ridge_model` is spliced into the network, is the only modification.

This completes the export process.

> **What's going on here?** MLJ's machine interface is built atop a more primitive *[model](adding_new_models.md)* interface, implemented for each algorithm. Each supervised model type (eg, `RidgeRegressor`) requires model `fit` and `predict` methods, which are called by the corresponding machine `fit!` and `predict` methods. We don't need to define a  model `predict` method here because MLJ provides a fallback which simply calls the node returned by `fit` on the data supplied: `MLJ.predict(model::Supervised{Node}, Xnew) = yhat(Xnew)`.

Let's now let's wrap our composite model as a tuned model and evaluate on the Boston dataset:


```julia
task = load_boston()
X, y = task()
train, test = partition(eachindex(y), 0.7)
wrapped_model = WrappedRidge(ridge_model)
```




    # [0m[1mWrappedRidge @ 1…44[22m: 
    ridge_model             =>   [0m[1mRidgeRegressor{Float64} @ 1…83[22m
    





```julia
params(wrapped_model)
```




    (ridge_model = (target_type = Float64, lambda = 0.01),)




```julia
nested_ranges = (ridge_model = (lambda = range(ridge_model, :lambda, lower=0.1, upper=100.0, scale=:log10),),)
```




    (ridge_model = (lambda = [0m[1mNumericRange @ 3…60[22m,),)




```julia
tuned_wrapped_model = TunedModel(model=wrapped_model, tuning=Grid(resolution=20),
resampling=CV(), measure=rms, nested_ranges=nested_ranges);
```


```julia
tuned_wrapped = machine(tuned_wrapped_model, X, y)
evaluate!(tuned_wrapped, resampling=Holdout(fraction_train=0.7), measure=rms, verbosity=0) |> mean  # nested resampling estimate
```




    6.88236977264247


