using BenchmarkDotNet.Attributes;
using MachineLearning.Domain.Activation;
using System.Buffers;

namespace MachineLearning.Benchmarks;

public class ActivationBenchmarks
{

    IActivationMethod ActivationFunction = SigmoidActivation.Instance;
    IActivationMethod NewActivationFunction = SoftmaxActivation.Instance;


    double[] InputArray = [];
    Vector input = default!;
    Vector result = default!;

    //[Params(128, 512, 2048)]
    public int size = 512;

    [GlobalSetup]
    public void Setup()
    {
        InputArray = Enumerable.Range(0, size).Select(n => Random.Shared.NextDouble()).ToArray();
        input = Vector.Of(InputArray);
        result = Vector.Create(size);
    }


    [Benchmark(Baseline = true)]
    public void Activate_Current()
    {
        ActivationFunction.Activate(input, result);
    }

    [Benchmark]
    public void Activate_Simd()
    {
        NewActivationFunction.Activate(input, result);
    }

    [Benchmark]
    public void Derivative_Current()
    {
        ActivationFunction.Derivative(input, result);
    }

    [Benchmark]
    public void Derivative_Simd()
    {
        NewActivationFunction.Derivative(input, result);
    }
}