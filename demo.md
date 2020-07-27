```
dim_red_ = PCA()
dim_red = machine(dim_red, X)
fit!(dim_red)
Xsmall = transform(dim_red, X)
```


```julia
classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
fit!(classifier)
ŷ = predict(classifier, Xsmall)
```

```julia
dim_red_ = PCA()
dim_red = machine(dim_red, X)
fit!(dim_red)
Xsmall = transform(dim_red, X)

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
fit!(classifier)
ŷ = predict(classifier, Xsmall)
```

```julia
dim_red_ = PCA()
dim_red = machine(dim_red, X)
Xsmall = transform(dim_red, X)

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
ŷ = predict(classifier, Xsmall)
```

```julia
X = source(X)
y = source(y)

dim_red_ = PCA()
dim_red = machine(dim_red, X)
Xsmall = transform(dim_red, X)

classifier_ = SVC()
classifier = machine(classifier_, Xsmall, y)
ŷ = predict(classifier, Xsmall)
```

```julia
fit!(ŷ)
ŷ()
```

```julia




## Building a composite model:

### Method 1: Compact syntax (but not generalizable):


```julia
composite = @pipeline pca svc

composite_ = machine(composite, X, y)
evaluate!(composite_, measure=misclassification_rate)
```

### Method 2: Just write the math (and test as you build):


```julia
Xraw = X;
yraw = y;

X = source(Xraw)
y = source(yraw)

pca_ = machine(pca, X)
Xsmall = transform(pca_, X)

svc_ = machine(svc, Xsmall, y)
yhat = predict(svc_, Xsmall)

yhat(rows=3:4)

fit!(yhat)

yhat(rows=3:4)
```


```julia
svc.gamma = 0.1
fit!(yhat)

yhat(rows=3:4)
```


```julia
svc.num_round = 10
fit!(yhat)
```


```julia
yhat(rows=3:4)
```

# Exporting network as stand-alone re-usable model:


```julia
composite = @from_network Composite(pca=pca, svc=svc) <= (X, y, yhat)
show(composite, 2)
```


```julia
composite_ = machine(composite, Xraw, yraw)
evaluate!(composite_, measure=misclassification_rate)
```

*This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
