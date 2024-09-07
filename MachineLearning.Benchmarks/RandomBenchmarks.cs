
namespace MachineLearning.Benchmarks;

//[ShortRunJob]
public class RandomBenchmarks
{
    public Vector left = default!;
    public Vector right = default!;
    public Matrix result = default!;

    [Params(1024)]
    public int Count = 5;

    [GlobalSetup]
    public void Setup()
    {
        left = Vector.Create(Count);
        right = Vector.Create(Count);
        result = Matrix.CreateSquare(Count);

        left.MapInPlace(_ => Random.Shared.NextDouble());
        right.MapInPlace(_ => Random.Shared.NextDouble());
    }

    [Benchmark(Baseline = true)]
    public void Multiply_Old()
    {
        VectorHelper.MultiplyToMatrixSimd(left, right, result);
    }

    [Benchmark]
    public void Multiply_New()
    {
        VectorHelper.MultiplyToMatrix(left, right, result);
    }
}
