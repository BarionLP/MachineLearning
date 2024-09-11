namespace MachineLearning.Benchmarks;

//[SimpleJob]
public class MatrixOperationsBenchmark
{
    private Matrix matrix1 = default!;
    private Matrix matrix2 = default!;
    private Vector rowVector = default!;
    private Vector columnVector = default!;
    private Matrix resultMatrix = default!;
    //private Matrix resultVector = default!;

    //[Params(16, 128, 512)]
    public int Rows { get; set; } = 128;

    //[Params(16, 128, 512)]
    public int Columns { get; set; } = 128;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(69);
        matrix1 = Matrix.Of(Rows, Columns, Enumerable.Range(0, Rows * Columns).Select(n => random.NextDouble()).ToArray());
        matrix2 = Matrix.Of(Rows, Columns, Enumerable.Range(0, Rows * Columns).Select(n => random.NextDouble()).ToArray());
        rowVector = Vector.Of(Enumerable.Range(0, Rows).Select(n => random.NextDouble()).ToArray());
        columnVector = Vector.Of(Enumerable.Range(0, Columns).Select(n => random.NextDouble()).ToArray());
        resultMatrix = Matrix.Create(Rows, Columns);
    }

    [Benchmark(Baseline = true)]
    public void VectorMultiply_Loop()
    {
        matrix1.MultiplyTo(columnVector, rowVector);
    }

    //[Benchmark]
    //public void VectorMultiply_Simd()
    //{
    //    MatriMultiplySimd(columnVector, rowVector);
    //}
}
