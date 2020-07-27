using MLJ

models()

using RDatasets
iris = dataset("datasets", "iris"); # a DataFrame
first(iris, 3)

task = supervised(data=iris,
                  target=:Species,
                  is_probabilistic=true)

models(task)

                  
