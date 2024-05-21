using BenchmarkDotNet.Attributes;
using MachineLearning.Domain.Activation;
using System.Buffers;

namespace Simple.Benchy;

public class ActivationBenchmarks {

    IActivationMethod ActivationFunction = SigmoidActivation.Instance;

    double[] InputArray = [];
    Vector InputVector = default!;

    //[Params(128, 512, 2048)]
    public int size = 512;

    [GlobalSetup]
    public void Setup() {
        InputArray = Enumerable.Range(0, size).Select(n=> Random.Shared.NextDouble()).ToArray();
        InputVector = Vector.Of(InputArray);
    }


    [Benchmark(Baseline = true)]
    public Vector SigmoidActivation_Current() {
        var activationDerivatives = ActivationFunction.Derivative(InputVector);
        return activationDerivatives;
    }
}