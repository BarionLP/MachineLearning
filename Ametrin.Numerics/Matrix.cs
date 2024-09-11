using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;

namespace Ametrin.Numerics;

// must be a row major continuous chunk of memory for current simd to work 
public interface Matrix
{
    public static readonly Matrix Empty = new MatrixFlat(0, 0, Vector.Empty);

    public int RowCount { get; }
    public int ColumnCount { get; }
    public int FlatCount { get; }
    public ref Weight this[int row, int column] { get; }
    public ref Weight this[nuint flatIndex] { get; }

    public Span<Weight> AsSpan();

    public static Matrix CreateSquare(int size) => Create(size, size);
    public static Matrix Create(int rowCount, int columnCount) => new MatrixFlat(rowCount, columnCount, Vector.Create(rowCount * columnCount));
    public static Matrix Of(int rowCount, int columnCount, double[] storage) => Of(rowCount, columnCount, Vector.Of(storage));
    public static Matrix Of(int rowCount, int columnCount, Vector storage)
    {
        if (storage.Count != columnCount * rowCount)
        {
            throw new ArgumentException("storage size does not match specified dimensions");
        }

        return new MatrixFlat(rowCount, columnCount, storage);
    }

    public static Matrix OfSize(Matrix template) => Create(template.RowCount, template.ColumnCount);
}

internal readonly struct MatrixFlat(int rowCount, int columnCount, Vector storage) : Matrix
{
    public Vector Storage { get; } = storage;
    public int RowCount { get; } = rowCount;
    public int ColumnCount { get; } = columnCount;
    public int FlatCount => Storage.Count;

    public ref Weight this[int row, int column] => ref Storage[GetFlatIndex(row, column)];
    public ref Weight this[nuint flatIndex] => ref Storage[flatIndex];
    public Span<Weight> AsSpan() => Storage.AsSpan();


    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Matrix ({RowCount}x{ColumnCount}):");
        for (int i = 0; i < RowCount; i++)
        {
            for (int j = 0; j < ColumnCount; j++)
            {
                sb.Append(this[i, j].ToString("F2")).Append(' ');
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    internal int GetFlatIndex(int row, int column)
    {
#if DEBUG
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, RowCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, ColumnCount);
#endif

        return row * ColumnCount + column;
    }
}

internal readonly struct TensorLayerReference(int layerIndex, Tensor tensor) : Matrix
{
    private readonly int _startIndex = tensor.RowCount * tensor.ColumnCount * layerIndex;
    private readonly Tensor _tensor = tensor;

    public int RowCount => _tensor.RowCount;
    public int ColumnCount => _tensor.ColumnCount;
    public int FlatCount => RowCount * FlatCount;

    public ref Weight this[int row, int column] => ref AsSpan()[row * ColumnCount + column];
    public ref Weight this[nuint index] => ref AsSpan()[(int)index];

    public Span<Weight> AsSpan() => _tensor.AsSpan().Slice(_startIndex, FlatCount);

    //public override string ToString()
    //{
    //    var builder = new StringBuilder("[");
    //    var data = AsSpan();
    //    for(int i = 0; i < data.Length; i++)
    //    {
    //        if(i > 0)
    //            builder.Append(' ');
    //        builder.Append(data[i].ToString("F2"));
    //    }
    //    builder.Append(']');
    //    return builder.ToString();
    //}
}


public static class MatrixHelper
{
    public static Vector Multiply(this Matrix matrix, Vector vector)
    {
        var result = Vector.Create(matrix.RowCount);
        Multiply(matrix, vector, result);
        return result;
    }

    public static void Multiply(this Matrix matrix, Vector vector, Vector result)
    {
        Debug.Assert(vector.Count == matrix.ColumnCount);
        Debug.Assert(result.Count == matrix.RowCount);

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());

        var mdSize = (nuint)SimdVector.Count;
        var rowCount = (nuint)matrix.RowCount;
        var columnCount = (nuint)matrix.ColumnCount;

        for (nuint row = 0; row < rowCount; row++)
        {
            nuint rowOffset = row * columnCount;
            nuint column = 0;
            var aggregator = SimdVector.Zero;
            for (; column + mdSize <= columnCount; column += mdSize)
            {
                var matrixVec = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                var vectorVec = SimdVectorHelper.LoadUnsafe(ref vectorPtr, column);
                aggregator += matrixVec * vectorVec;
            }

            ref var sum = ref result[row];
            sum = SimdVectorHelper.Sum(aggregator);

            for (; column < columnCount; column++)
            {
                sum += matrix[rowOffset + column] * vector[column];
            }
        }
    }

    public static void MultiplyRowwise(this Matrix left, Matrix right, Matrix result)
    {
        Debug.Assert(left.RowCount == result.RowCount);

        for (int rowIndex = 0; rowIndex < left.RowCount; rowIndex++)
        {
            var row = left.RowRef(rowIndex);
            var resultRow = result.RowRef(rowIndex);
            right.Multiply(row, resultRow);
        }
    }

    #region ChatGPT
    public static Vector MultiplyTransposed(this Matrix matrix, Vector vector)
    {
        var result = Vector.Create(vector.Count);
        MultiplyTransposed(matrix, vector, result);
        return result;
    }
    public static void MultiplyTransposed(this Matrix matrix, Vector vector, Vector result)
    {
        Debug.Assert(matrix.RowCount == vector.Count);
        Debug.Assert(matrix.ColumnCount == result.Count);

        //result.ResetZero(); // Reset result vector to zero

        for (int col = 0; col < matrix.ColumnCount; col++)
        {
            for (int row = 0; row < matrix.RowCount; row++)
            {
                result[col] += matrix[row, col] * vector[row]; // Multiply matrix transpose element by vector element
            }
        }
    }

    public static void SoftMaxGradientInPlace(Vector gradient, Vector softmax)
    {
        Debug.Assert(gradient.Count == softmax.Count);

        for (int i = 0; i < softmax.Count; i++)
        {
            Weight si = softmax[i];
            gradient[i] *= si * (1 - si); // Gradient of softmax for the diagonal terms

            for (int j = 0; j < softmax.Count; j++)
            {
                if (i != j)
                {
                    gradient[i] -= si * softmax[j] * gradient[j]; // Gradient of softmax for the off-diagonal terms
                }
            }
        }
    }

    public static void MultiplyTransposeWithGradient(Vector gradient, Vector inputRow, Matrix weightGradient)
    {
        Debug.Assert(gradient.Count == weightGradient.RowCount);
        Debug.Assert(inputRow.Count == weightGradient.ColumnCount);

        for (int i = 0; i < weightGradient.RowCount; i++)
        {
            for (int j = 0; j < weightGradient.ColumnCount; j++)
            {
                weightGradient[i, j] += gradient[i] * inputRow[j]; // Compute and accumulate the weight gradient
            }
        }
    }
    #endregion

    public static void MapToSelf(this Matrix matrix, Func<Weight, Weight> map) => matrix.Map(map, matrix);
    public static Matrix Map(this Matrix matrix, Func<Weight, Weight> map)
    {
        var result = Matrix.Create(matrix.RowCount, matrix.ColumnCount);
        matrix.Map(map, result);
        return result;
    }
    public static void Map(this Matrix matrix, Func<Weight, Weight> map, Matrix result)
    {
        AssertCountEquals(matrix, result);
        SpanOperations.Map(matrix.AsSpan(), result.AsSpan(), map);
    }

    public static void MapInFirst(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map) => matrices.Map(matrices.a, map);
    public static Matrix Map(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map)
    {
        var result = Matrix.Create(matrices.a.RowCount, matrices.a.ColumnCount);
        matrices.Map(result, map);
        return result;
    }
    public static void Map(this (Matrix a, Matrix b) matrices, Matrix result, Func<Weight, Weight, Weight> map)
    {
        AssertCountEquals(matrices.a, matrices.b, result);
        SpanOperations.Map(matrices.a.AsSpan(), matrices.b.AsSpan(), result.AsSpan(), map);
    }
    public static Matrix Map(this (Matrix a, Matrix b, Matrix c) matrices, Func<Weight, Weight, Weight, Weight> map)
    {
        var result = Matrix.OfSize(matrices.a);
        matrices.Map(result, map);
        return result;
    }
    public static void Map(this (Matrix a, Matrix b, Matrix c) matrices, Matrix result, Func<Weight, Weight, Weight, Weight> map)
    {
        AssertCountEquals(matrices.a, matrices.b, result);
        SpanOperations.Map(matrices.a.AsSpan(), matrices.b.AsSpan(), matrices.c.AsSpan(), result.AsSpan(), map);
    }

    public static void AddInPlace(this Matrix left, Matrix right)
    {
        Add(left, right, left);
    }
    public static Matrix Add(this Matrix left, Matrix right)
    {
        var result = Matrix.Create(left.RowCount, left.ColumnCount);
        Add(left, right, result);
        return result;
    }

    public static void Add(this Matrix left, Matrix right, Matrix result)
    {
        AssertCountEquals(left, right, result);
        TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    public static void SubtractInPlace(this Matrix left, Matrix right)
    {
        Subtract(left, right, left);
    }
    public static Matrix Subtract(this Matrix left, Matrix right)
    {
        var result = Matrix.Create(left.RowCount, left.ColumnCount);
        Subtract(left, right, result);
        return result;
    }

    public static void Subtract(this Matrix left, Matrix right, Matrix result)
    {
        AssertCountEquals(left, right, result);
        TensorPrimitives.Subtract(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    public static Span<double> RowSpan(this Matrix matrix, int rowIndex) => matrix.AsSpan().Slice(rowIndex * matrix.ColumnCount, matrix.ColumnCount);
    public static Vector RowRef(this Matrix matrix, int rowIndex) => new MatrixRowReference(rowIndex, matrix);

    public static Matrix Copy(this Matrix matrix)
    {
        var copy = Matrix.Create(matrix.RowCount, matrix.ColumnCount);
        matrix.AsSpan().CopyTo(copy.AsSpan());
        return copy;
    }

    public static void ResetZero(this Matrix matrix)
    {
        matrix.AsSpan().Clear();
    }

    const string MATRIX_COUNT_MISMATCH = "matrices must match in size!";

    [Conditional("DEBUG")]
    private static void AssertCountEquals(Matrix a, Matrix b)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount, MATRIX_COUNT_MISMATCH);
    }

    [Conditional("DEBUG")]
    private static void AssertCountEquals(Matrix a, Matrix b, Matrix c)
    {
        Debug.Assert(a.RowCount == b.RowCount && b.RowCount == c.RowCount &&
                     a.ColumnCount == b.ColumnCount && b.ColumnCount == c.ColumnCount, MATRIX_COUNT_MISMATCH);
    }
}