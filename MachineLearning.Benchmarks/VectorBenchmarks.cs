using System.Numerics.Tensors;

namespace MachineLearning.Benchmarks;

public class VectorBenchmarks
{
    private Vector Vector1 = Vector.Empty;

    [Params(1023, 1_000_000)]
    public int Size = 4;

    [GlobalSetup]
    public void Setup()
    {
        Vector1 = Vector.Create(Size);
    }

    [Benchmark(Baseline = true)]
    public Weight MySum() => Vector1.Sum();

    [Benchmark]
    public Weight NativeSum() => TensorPrimitives.Sum(Vector1.AsSpan());
}
