namespace MachineLearning.Benchmarks;

//[ShortRunJob]
public class RandomBenchmarks
{
    public Matrix left = default!;
    public Vector right = default!;
    public Vector result = default!;

    [Params(1024)]
    public int Count = 5;

    [GlobalSetup]
    public void Setup()
    {
        left = Matrix.CreateSquare(Count);
        right = Vector.Create(Count);
        result = Vector.Create(Count);

        left.MapToSelf(_ => Random.Shared.NextDouble());
        right.MapToSelf(_ => Random.Shared.NextDouble());
    }

    [Benchmark(Baseline = true)]
    public void Multiply_Old()
    {
        //MatrixHelper.MultiplySimd(left, right, result);
    }

    [Benchmark]
    public void Multiply_New()
    {
        MatrixHelper.MultiplyTo(left, right, result);
    }
}
