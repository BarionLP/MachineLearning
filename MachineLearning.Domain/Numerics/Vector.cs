using System;
using System.Data.Common;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace MachineLearning.Domain.Numerics;

public interface Vector
{
    public static readonly Vector Empty = new VectorSimple(0, []);
    public int Count { get; }
    public ref Weight this[int index] { get; }
    public ref Weight this[nuint index] { get; }
    public Span<Weight> this[int index, int count] { get; }
    public Span<Weight> AsSpan();

    public static Vector Create(int count) => new VectorSimple(count, new Weight[count]);
    public static Vector Of(Weight[] array) => new VectorSimple(array.Length, array);
}

// stores its own count to allow longer arrays from ArrayPool as storage
internal readonly struct VectorSimple(int count, Weight[] storage) : Vector
{
    private readonly Weight[] _storage = storage;
    public int Count { get; } = count;
    public ref Weight this[int index] => ref _storage[index];
    public ref Weight this[nuint index] => ref _storage[index];
    public Span<Weight> this[int index, int count] => new(_storage, index, count); // has built-in bound checks
    public Span<Weight> AsSpan() => new(_storage, 0, Count);
    public override string ToString() => $"[{string.Join(' ', _storage.Select(d => d.ToString("F2")))}]";
}

internal readonly struct MatrixRowReference(int _rowIndex, Matrix _matrix) : Vector
{
    private readonly int _startIndex = _rowIndex * _matrix.ColumnCount;
    private readonly Matrix _matrix = _matrix;

    public ref double this[int index] => ref AsSpan()[index];

    public ref double this[nuint index] => ref AsSpan()[(int)index];

    public Span<double> this[int index, int count] => AsSpan().Slice(index, count);

    public int Count => _matrix.ColumnCount;

    public Span<double> AsSpan() => _matrix.AsSpan().Slice(_startIndex, _matrix.ColumnCount);

    public override string ToString()
    {
        var builder = new StringBuilder("[");
        var data = AsSpan();
        for (int i = 0; i < data.Length; i++)
        {
            if (i > 0) builder.Append(' ');
            builder.Append(data[i].ToString("F2"));
        }
        builder.Append(']');
        return builder.ToString();
    }
}

public static class VectorHelper
{
    public static Weight Sum(this Vector vector)
    {
        // return TensorPrimitives.Sum<double>(vector.AsSpan()); // seems to be slower
        ref var ptr = ref MemoryMarshal.GetReference(vector.AsSpan());
        nuint length = (nuint)vector.Count;

        var accumulator = SimdVector.Zero;
        nuint index = 0;

        var limit = length - (nuint)SimdVector.Count;
        for (; index < limit; index += (nuint)SimdVector.Count)
        {
            accumulator += SimdVectorHelper.LoadUnsafe(ref ptr, index);
        }

        var result = SimdVectorHelper.Sum(accumulator);

        for (; index < length; index++)
        {
            result += Unsafe.Add(ref ptr, index);
        }

        return result;
    }

    public static Weight Dot(this Vector left, Vector right)
    {
        AssertCountEquals(left, right);
        return TensorPrimitives.Dot<double>(left.AsSpan(), right.AsSpan());
    }

    public static void PointwiseExpInPlace(this Vector vector) => PointwiseExp(vector, vector);
    public static void PointwiseExp(this Vector vector, Vector destination)
    {
        TensorPrimitives.Exp(vector.AsSpan(), destination.AsSpan());
    }

    public static void SoftMaxInPlace(this Vector vector) => SoftMax(vector, vector);
    public static Vector SoftMax(this Vector vector) {
        var result = Vector.Create(vector.Count);
        SoftMax(vector, result);
        return result;
    }

    public static void SoftMax(this Vector vector, Vector destination)
    {
        vector.PointwiseExp(destination);
        var sum = destination.Sum();
        destination.DivideInPlace(sum);
        
        // was slower in .net 9.preview.7
        //TensorPrimitives.SoftMax(vector.AsSpan(), destination.AsSpan());
    }

    public static void MapInPlace(this Vector vector, Func<Weight, Weight> map) => vector.Map(map, vector);
    public static Vector Map(this Vector vector, Func<Weight, Weight> map)
    {
        var result = Vector.Create(vector.Count);
        vector.Map(map, result);
        return result;
    }
    public static void Map(this Vector vector, Func<Weight, Weight> map, Vector result)
    {
        AssertCountEquals(vector, result);
        SpanOperations.Map(vector.AsSpan(), result.AsSpan(), map);
    }

    public static void MapInPlace(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap) => vector.Map(simdMap, fallbackMap, vector);
    public static void Map(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap, Vector result)
    {
        AssertCountEquals(vector, result);
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint)SimdVector.Count;
        var totalSize = (nuint)vector.Count;

        nuint index = 0;
        for (; index + mdSize <= totalSize; index += mdSize)
        {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe(simdMap.Invoke(simdVector), ref resultPtr, index);
        }

        for (; index < totalSize; index++)
        {
            result[index] = fallbackMap.Invoke(vector[index]);
        }
    }

    public static void MapInFirst(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map) => vectors.Map(map, vectors.a);
    public static Vector Map(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map)
    {
        var result = Vector.Create(vectors.a.Count);
        vectors.Map(map, result);
        return result;
    }
    public static void Map(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map, Vector result)
    {
        AssertCountEquals(vectors.a, vectors.b, result);
        for (int i = 0; i < vectors.a.Count; i++)
        {
            result[i] = map.Invoke(vectors.a[i], vectors.b[i]);
        }
    }

    public static void PointwiseMultiplyInPlace(this Vector left, Vector right) => left.PointwiseMultiply(right, left);
    public static Vector PointwiseMultiply(this Vector left, Vector right)
    {
        var result = Vector.Create(left.Count);
        left.PointwiseMultiply(right, result);
        return result;
    }
    public static void PointwiseMultiply(this Vector left, Vector right, Vector result)
    {
        AssertCountEquals(left, right, result);
        TensorPrimitives.Multiply(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    public static Vector Multiply(this Vector vector, Matrix matrix)
    {
        var result = Vector.Create(matrix.ColumnCount);
        vector.Multiply(matrix, result);
        return result;
    }

    public static void Multiply(this Vector vector, Matrix matrix, Vector result)
    {
        //Story time: swapping loops increased performance by 85 % because of increased cache hits (before simd impl)
        Debug.Assert(vector.Count == matrix.RowCount);
        Debug.Assert(result.Count == matrix.ColumnCount);

        result.ResetZero();

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        var rowCount = (nuint) matrix.RowCount;
        var columnCount = (nuint) matrix.ColumnCount;


        for(nuint row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(vector[row]);
            var rowOffset = row * columnCount;
            nuint column = 0;
            for(; column <= columnCount - mdSize; column += mdSize)
            {
                var resultValues = SimdVectorHelper.LoadUnsafe(ref resultPtr, column);
                var matrixValues = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                resultValues += rowValue * matrixValues;

                SimdVectorHelper.StoreUnsafe(resultValues, ref resultPtr, column);
            }

            for(; column < columnCount; column++)
            {
                result[column] += vector[row] * matrix[rowOffset + column];
            }
        }
    }

    public static Matrix MultiplyToMatrix(Vector rowVector, Vector columnVector)
    {
        var result = Matrix.Create(rowVector.Count, columnVector.Count);
        MultiplyToMatrix(rowVector, columnVector, result);
        return result;
    }

    public static void MultiplyToMatrix(Vector rowVector, Vector columnVector, Matrix result)
    {
        Debug.Assert(rowVector.Count == result.RowCount);
        Debug.Assert(columnVector.Count == result.ColumnCount);

        ref var columnPtr = ref MemoryMarshal.GetReference(columnVector.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());

        var rowCount = (nuint) rowVector.Count;
        var columnCount = (nuint) columnVector.Count;

        nuint mdSize = (nuint) SimdVector.Count;

        for(nuint row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(rowVector[row]);
            var rowOffset = row * columnCount;

            nuint column = 0;
            for(; column <= columnCount - mdSize; column += mdSize)
            {
                var columnValues = SimdVectorHelper.LoadUnsafe(ref columnPtr, column);
                var resultValues = rowValue * columnValues;

                SimdVectorHelper.StoreUnsafe(resultValues, ref resultPtr, rowOffset + column);
            }

            for(; column < columnCount; column++)
            {
                result[rowOffset + column] = rowVector[row] * columnVector[column];
            }
        }
    }

    public static void MultiplyToMatrixSimd(Vector rowVector, Vector columnVector, Matrix result)
    {
        Debug.Assert(rowVector.Count == result.RowCount);
        Debug.Assert(columnVector.Count == result.ColumnCount);

        int rowCount = rowVector.Count;
        int colCount = columnVector.Count;

        //if (rowCount == 0 || colCount == 0 || rowCount * colCount != result.FlatCount)
        //    throw new ArgumentException("Invalid dimensions for the vectors or result matrix.");

        int mdSize = SimdVector.Count;

        for (int row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(rowVector[row]);

            int col;
            for (col = 0; col <= colCount - mdSize; col += mdSize)
            {
                var colValues = new SimdVector(columnVector.AsSpan().Slice(col, SimdVector.Count));
                var resultValues = rowValue * colValues;

                for (int i = 0; i < mdSize; i++)
                {
                    result[row, col + i] = resultValues[i];
                }
            }

            for (; col < colCount; col++)
            {
                result[row, col] = rowVector[row] * columnVector[col];
            }
        }
    }


    public static void MultiplyInPlace(this Vector vector, Weight factor) => Multiply(vector, factor, vector);
    public static Vector Multiply(this Vector vector, Weight factor)
    {
        var result = Vector.Create(vector.Count);
        Multiply(vector, factor, result);
        return result;
    }
    public static void Multiply(this Vector vector, Weight factor, Vector result)
    {
        AssertCountEquals(vector, result);
        TensorPrimitives.Multiply(vector.AsSpan(), factor, result.AsSpan());
    }

    public static void DivideInPlace(this Vector vector, Weight divisor) => Divide(vector, divisor, vector);
    public static Vector Divide(this Vector vector, Weight divisor)
    {
        var result = Vector.Create(vector.Count);
        Divide(vector, divisor, result);
        return result;
    }
    public static void Divide(this Vector vector, Weight divisor, Vector result)
    {
        Multiply(vector, 1 / divisor, result);
    }

    public static void AddInPlace(this Vector left, Vector right) => Add(left, right, left);
    public static Vector Add(this Vector left, Vector right)
    {
        var result = Vector.Create(left.Count);
        Add(left, right, result);
        return result;
    }

    public static void Add(this Vector left, Vector right, Vector result)
    {
        AssertCountEquals(left, right, result);
        TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    public static void SubtractPointwiseInPlace(this Vector left, Weight right) => SubtractPointwise(left, right, left);
    public static Vector SubtractPointwise(this Vector left, Weight right)
    {
        var result = Vector.Create(left.Count);
        SubtractPointwise(left, right, result);
        return result;
    }
    public static void SubtractPointwise(this Vector left, Weight right, Vector result)
    {
        AssertCountEquals(left, result);
        TensorPrimitives.Subtract(left.AsSpan(), right, result.AsSpan());
    }
    
    public static void SubtractInPlace(this Vector left, Vector right) => Subtract(left, right, left);
    public static Vector Subtract(this Vector left, Vector right)
    {
        var result = Vector.Create(left.Count);
        Subtract(left, right, result);
        return result;
    }
    public static void Subtract(this Vector left, Vector right, Vector result)
    {
        AssertCountEquals(left, right, result);
        TensorPrimitives.Subtract(left.AsSpan(), right.AsSpan(), result.AsSpan());
    }

    public static int MaximumIndex(this Vector vector)
    {
        var maxIndex = 0;
        for (int i = 1; i < vector.Count; i++)
        {
            if (vector[i] > vector[maxIndex])
            {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static Weight Max(this Vector vector) => TensorPrimitives.Max<Weight>(vector.AsSpan());

    //TODO: test
    public static Vector Copy(this Vector vector)
    {
        var copy = Vector.Create(vector.Count);
        vector.AsSpan().CopyTo(copy.AsSpan());
        return copy;
    }

    //TODO: test
    public static void CopyTo(this Vector vector, Vector target)
    {
        AssertCountEquals(vector, target);
        vector.AsSpan().CopyTo(target.AsSpan());
    }

    public static void ResetZero(this Vector vector) => vector.AsSpan().Clear();

    const string VECTORS_MUST_MATCH = "Vectors must match in size";
    [Conditional("DEBUG")]
    private static void AssertCountEquals(Vector a, Vector b)
    {
        Debug.Assert(a.Count == b.Count, VECTORS_MUST_MATCH);
    }
    [Conditional("DEBUG")]
    private static void AssertCountEquals(Vector a, Vector b, Vector c)
    {
        Debug.Assert(a.Count == b.Count && a.Count == b.Count, VECTORS_MUST_MATCH);
    }
}
