namespace MachineLearning.Benchmarks;

//[SimpleJob]
public class MatrixOperationsBenchmark
{
    public Matrix matrix1 = default!;
    public Matrix matrix2 = default!;
    public Vector rowVector = default!;
    public Vector columnVector = default!;
    public Matrix resultMatrix = default!;
    //private Matrix resultVector = default!;

    //[Params(16, 128, 512)]
    public int Rows { get; set; } = 6;

    //[Params(16, 128, 512)]
    public int Columns { get; set; } = 6;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(69);
        matrix1 = Matrix.Of(Rows, Columns, [.. Enumerable.Range(0, Rows * Columns).Select(n => random.NextSingle())]);
        matrix2 = Matrix.Of(Rows, Columns, [.. Enumerable.Range(0, Rows * Columns).Select(n => random.NextSingle())]);
        rowVector = Vector.Of([.. Enumerable.Range(0, Rows).Select(n => random.NextSingle())]);
        columnVector = Vector.Of([.. Enumerable.Range(0, Columns).Select(n => random.NextSingle())]);
        resultMatrix = Matrix.Create(Rows, Columns);
    }

    //these are not the same!!!

    [Benchmark(Baseline = true)]
    public void MultiplyRowwise()
    {
        matrix1.MultiplyRowwiseTo(matrix2, resultMatrix);
    }

    [Benchmark]
    public void Multiply()
    {
        Multiply(matrix1, matrix2, resultMatrix);
    }

    public static Matrix Multiply(Matrix matrixA, Matrix matrixB, Matrix resultMatrix)
    {
        if (matrixA.ColumnCount != matrixB.RowCount)
            throw new InvalidOperationException("Matrix dimensions do not match for multiplication.");

        for (int i = 0; i < matrixA.RowCount; i++)
        {
            for (int j = 0; j < matrixB.ColumnCount; j++)
            {
                Weight sum = 0;
                for (int k = 0; k < matrixA.ColumnCount; k++)
                {
                    sum += matrixA[i, k] * matrixB[k, j];
                }
                resultMatrix[i, j] = sum;
            }
        }

        return resultMatrix;
    }
}
