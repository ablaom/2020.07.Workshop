using DataFrames
using CSV
using Random
using Plots
import MLJ

pyplot(size=(460,200))


fit(x, y, n) = collect(v ^ p for v in x, p in 0:n) \ y
predict(result, xnew) = sum([result[j]*xnew^(j-1) for j in eachindex(result)])
predict(result, Xnew::AbstractVector) = map(xnew->predict(result, xnew), Xnew)

x = 0:0.5:12
train = 1:2:25
test = 2:2:25


Random.seed!(1234)
y = 12 .+ 4*(sin.(2pi*x/24) + 0.2*randn(length(x)))
function yhat(n)
    result = fit(x[train],y[train],n)
    return x -> predict(result, x)
end

function curve(n)
    x = range(0, 12, length=100)
    y = yhat(n).(x)
    plot!(x, y, label="degree = $n")
end

test_error(n) = MLJ.rms(y[test], yhat(n).(x[test]))
train_error(n) = MLJ.rms(y[train], yhat(n).(x[train]))

plt1 = scatter(x[train], y[train], ms = 5, label="training data", ylim=(7,21),
               xlab="time (hours)", ylab="temperature (deg C)")
savefig("overfitting1.png")
plt2 = curve(1)
savefig("overfitting2.png")
plt3 = curve(3)
savefig("overfitting3.png")
plt4 = curve(12)
savefig("overfitting4.png")
plt5 = scatter!(x[test], y[test], ms = 5, label="test data", markershape=:cross)
savefig("overfitting5.png")

plt6 = scatter(x[test], y[test], ms = 5, label="test data", markershape=:cross,
               ylim=(0,21), xlab="time (hours)", ylab="temperature (deg C)")
curve(1)
curve(3)
curve(12)
savefig("overfitting6.png")

argmin([test_error(n) for n in 1:12])

test_curve = [test_error(n) for n in 0:12]
train_curve = [train_error(n) for n in 0:12]
plt_performance = plot(0:12, hcat(test_curve, train_curve),
                       label=permutedims(["test error", "train_error"]),
                       xlab="polynomial degree",
                       ylab="RMS error")
savefig("performance.png")
