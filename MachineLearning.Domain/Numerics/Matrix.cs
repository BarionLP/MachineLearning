using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace MachineLearning.Domain.Numerics;

// must be a continuous chunk of memory for current simd to work 
public interface Matrix
{
    public int RowCount { get; }
    public int ColumnCount { get; }
    public int FlatCount { get; }
    public ref Weight this[int row, int column] { get; }
    public ref Weight this[nuint flatIndex] { get; }
    internal Vector Storage { get; }

    public Span<Weight> AsSpan();
    public Vector RowRef(int rowIndex);

    public static Matrix CreateSquare(int size) => Create(size, size);
    public static Matrix Create(int rowCount, int columnCount) => new MatrixFlat(rowCount, columnCount, Vector.Create(rowCount * columnCount));
    public static Matrix Of(int rowCount, int columnCount, double[] storage) => Of(rowCount, columnCount, Vector.Of(storage));
    public static Matrix Of(int rowCount, int columnCount, Vector storage)
    {
        if(storage.Count != columnCount * rowCount)
        {
            throw new ArgumentException("storage size does not match specified dimensions");
        }

        return new MatrixFlat(rowCount, columnCount, storage);
    }
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

    public Vector RowRef(int rowIndex) => new MatrixRowReference(rowIndex, this);

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Matrix ({RowCount}x{ColumnCount}):");
        for(int i = 0; i < RowCount; i++)
        {
            for(int j = 0; j < ColumnCount; j++)
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
        MultiplySimd(matrix, vector, result);

        //for(int row = 0; row < matrix.RowCount; row++) {
        //    result[row] = 0;
        //    for(int column = 0; column < matrix.ColumnCount; column++) {
        //        result[row] += matrix[row, column] * vector[column];
        //    }
        //}
    }

    public static void MultiplySimd(Matrix matrix, Vector vector, Vector result)
    {
        Debug.Assert(vector.Count == matrix.ColumnCount);
        Debug.Assert(result.Count == matrix.RowCount);

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());

        var mdSize = (nuint) SimdVector.Count;
        var rowCount = (nuint) matrix.RowCount;
        var columnCount = (nuint) matrix.ColumnCount;

        for(nuint row = 0; row < rowCount; row++)
        {
            nuint rowOffset = row * columnCount;
            Weight sum = 0;

            nuint column = 0;
            for(; column + mdSize <= columnCount; column += mdSize)
            {
                var matrixVec = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                var vectorVec = SimdVectorHelper.LoadUnsafe(ref vectorPtr, column);
                sum += SimdVectorHelper.Dot(matrixVec, vectorVec);
            }

            for(; column < columnCount; column++)
            {
                sum += matrix[row * columnCount + column] * vector[column];
            }

            result[row] = sum;
        }
    }

    public static void MultiplyRowwise(this Matrix left, Matrix right, Matrix result)
    {
        Debug.Assert(left.RowCount == result.RowCount);

        for(int rowIndex = 0; rowIndex < left.RowCount; rowIndex++)
        {
            var row = left.RowRef(rowIndex);
            var resultRow = result.RowRef(rowIndex);
            right.Multiply(row, resultRow);
        }
    }

    public static void MapInPlace(this Matrix matrix, Func<Weight, Weight> map) => matrix.Map(map, matrix);
    public static Matrix Map(this Matrix matrix, Func<Weight, Weight> map)
    {
        var result = Matrix.Create(matrix.RowCount, matrix.ColumnCount);
        matrix.Map(map, result);
        return result;
    }
    public static void Map(this Matrix matrix, Func<Weight, Weight> map, Matrix result)
    {
        AssertCountEquals(matrix, result);
        matrix.Storage.Map(map, result.Storage);
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

        (matrices.a.Storage, matrices.b.Storage).Map(map, result.Storage);
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
        
        left.Storage.Add(right.Storage, result.Storage);
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

        left.Storage.Subtract(right.Storage, result.Storage);
    }

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
    private static void AssertCountEquals(Matrix a, Matrix b)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount, MATRIX_COUNT_MISMATCH);
    }
    private static void AssertCountEquals(Matrix a, Matrix b, Matrix c)
    {
        Debug.Assert(a.RowCount == b.RowCount && b.RowCount == c.RowCount && 
                     a.ColumnCount == b.ColumnCount && b.ColumnCount == c.ColumnCount, MATRIX_COUNT_MISMATCH);
    }
}