using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Ametrin.Numerics;

public interface Vector
{
    public static readonly Vector Empty = new VectorSimple(0, []);
    public int Count { get; }
    public ref Weight this[int index] { get; }
    public ref Weight this[nuint index] { get; }
    public Span<Weight> Slice(int index, int count);
    public Span<Weight> AsSpan();

    public static Vector Create(int size) => new VectorSimple(size, new Weight[size]);
    public static Vector Of(Weight[] array) => new VectorSimple(array.Length, array);
    public static Vector Of(int size, Weight[] array)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(size, array.Length);
        return new VectorSimple(size, array);
    }
}

// stores its own count to allow longer arrays from ArrayPool as storage
internal readonly struct VectorSimple(int count, Weight[] storage) : Vector
{
    internal readonly Weight[] _storage = storage;
    public int Count { get; } = count;
    public ref Weight this[int index] => ref _storage[index];
    public ref Weight this[nuint index] => ref _storage[index];
    public Span<Weight> Slice(int index, int count) => new(_storage, index, count); // has built-in bound checks
    public Span<Weight> AsSpan() => new(_storage, 0, Count);
    public override string ToString() => $"[{string.Join(' ', _storage.Select(d => d.ToString("+#0.00;-#0.00")))}]";
}

internal readonly struct MatrixRowReference(int _rowIndex, Matrix _matrix) : Vector
{
    private readonly int _startIndex = _rowIndex * _matrix.ColumnCount;
    private readonly Matrix _matrix = _matrix;

    public ref double this[int index] => ref _matrix[_startIndex + index];

    public ref double this[nuint index] => ref _matrix[_startIndex + (int)index];

    public Span<Weight> Slice(int index, int count) => AsSpan().Slice(index, count);

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
        NumericsDebug.AssertSameDimensions(left, right);
        return TensorPrimitives.Dot<Weight>(left.AsSpan(), right.AsSpan());
    }

    public static Vector PointwiseExp(this Vector vector)
    {
        var destination = Vector.Create(vector.Count);
        PointwiseExpTo(vector, destination);
        return destination;
    }

    public static void PointwiseExpToSelf(this Vector vector) => PointwiseExpTo(vector, vector);
    public static void PointwiseExpTo(this Vector vector, Vector destination)
    {
        //var original = vector.CreateCopy();
        NumericsDebug.AssertSameDimensions(vector, destination);
        //TensorPrimitives.Exp(vector.AsSpan(), destination.AsSpan());

        for(int i = 0; i < vector.Count; i++)
        {   
            destination[i] = Math.Exp(vector[i]);
        }
    }

    public static void PointwiseLogToSelf(this Vector vector) => PointwiseLogTo(vector, vector);
    public static void PointwiseLogTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.Log(vector.AsSpan(), destination.AsSpan());
    }

    public static void SoftMaxToSelf(this Vector vector) => SoftMaxTo(vector, vector);
    public static Vector SoftMax(this Vector vector)
    {
        var destination = Vector.Create(vector.Count);
        SoftMaxTo(vector, destination);
        return destination;
    }

    public static void SoftMaxTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);

        var max = vector.Max();
        vector.SubtractPointwiseTo(max, destination);
        NumericsDebug.AssertValidNumbers(destination);
        var tmp = destination.PointwiseExp();
        var sum = tmp.Sum();
        if(sum is double.NegativeInfinity)
        {
            throw new ArgumentException();
        }
        tmp.DivideTo(sum, destination);

        NumericsDebug.AssertValidNumbers(tmp);

        // was slower in .net 9.preview.7
        //TensorPrimitives.SoftMax(vector.AsSpan(), destination.AsSpan());
    }

    public static void MapToSelf(this Vector vector, Func<Weight, Weight> map) => SpanOperations.MapTo(vector.AsSpan(), vector.AsSpan(), map);
    public static Vector Map(this Vector vector, Func<Weight, Weight> map)
    {
        var destination = Vector.Create(vector.Count);
        SpanOperations.MapTo(vector.AsSpan(), destination.AsSpan(), map);
        return destination;
    }
    public static void MapTo(this Vector vector, Func<Weight, Weight> map, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        SpanOperations.MapTo(vector.AsSpan(), destination.AsSpan(), map);
    }

    public static void MapToSelf(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap) => vector.MapTo(simdMap, fallbackMap, vector);
    public static void MapTo(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination.AsSpan());
        var mdSize = (nuint)SimdVector.Count;
        var totalSize = (nuint)vector.Count;

        nuint index = 0;
        for (; index + mdSize <= totalSize; index += mdSize)
        {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe(simdMap.Invoke(simdVector), ref destinationPtr, index);
        }

        for (; index < totalSize; index++)
        {
            destination[index] = fallbackMap.Invoke(vector[index]);
        }
    }

    public static void MapToFirst(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map) => vectors.MapTo(map, vectors.a);
    public static Vector Map(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map)
    {
        var destination = Vector.Create(vectors.a.Count);
        vectors.MapTo(map, destination);
        return destination;
    }
    public static void MapTo(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vectors.a, vectors.b, destination);
        SpanOperations.MapTo(vectors.a.AsSpan(), vectors.b.AsSpan(), destination.AsSpan(), map);
    }

    public static Vector Map(this (Vector a, Vector b, Vector c) vectors, Func<Weight, Weight, Weight, Weight> map)
    {
        var result = Vector.Create(vectors.a.Count);
        vectors.MapTo(map, result);
        return result;
    }
    public static void MapTo(this (Vector a, Vector b, Vector c) vectors, Func<Weight, Weight, Weight, Weight> map, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vectors.a, vectors.b, vectors.c, destination);
        SpanOperations.MapTo(vectors.a.AsSpan(), vectors.b.AsSpan(), vectors.c.AsSpan(), destination.AsSpan(), map);
    }

    public static void PointwiseMultiplyToSelf(this Vector left, Vector right) => left.PointwiseMultiplyTo(right, left);
    public static Vector PointwiseMultiply(this Vector left, Vector right)
    {
        var result = Vector.Create(left.Count);
        left.PointwiseMultiplyTo(right, result);
        return result;
    }
    public static void PointwiseMultiplyTo(this Vector left, Vector right, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.Multiply(left.AsSpan(), right.AsSpan(), destination.AsSpan());
    }

    public static Vector Multiply(this Vector vector, Matrix matrix)
    {
        var result = Vector.Create(matrix.ColumnCount);
        vector.MultiplyTo(matrix, result);
        return result;
    }

    public static void MultiplyTo(this Vector vector, Matrix matrix, Vector destination)
    {
        //Story time: swapping loops increased performance by 85 % because of increased cache hits (before simd impl)
        Debug.Assert(vector.Count == matrix.RowCount);
        Debug.Assert(destination.Count == matrix.ColumnCount);

        destination.ResetZero();

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(destination.AsSpan());
        var mdSize = (nuint)SimdVector.Count;
        var rowCount = (nuint)matrix.RowCount;
        var columnCount = (nuint)matrix.ColumnCount;


        for (nuint row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(vector[row]);
            var rowOffset = row * columnCount;
            nuint column = 0;
            for (; column <= columnCount - mdSize; column += mdSize)
            {
                var resultValues = SimdVectorHelper.LoadUnsafe(ref resultPtr, column);
                var matrixValues = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                resultValues += rowValue * matrixValues;

                SimdVectorHelper.StoreUnsafe(resultValues, ref resultPtr, column);
            }

            for (; column < columnCount; column++)
            {
                destination[column] += vector[row] * matrix[rowOffset + column];
            }
        }
    }

    public static Matrix MultiplyToMatrix(Vector rowVector, Vector columnVector)
    {
        var result = Matrix.Create(rowVector.Count, columnVector.Count);
        MultiplyToMatrixTo(rowVector, columnVector, result);
        return result;
    }

    public static void MultiplyToMatrixTo(Vector rowVector, Vector columnVector, Matrix destination)
    {
        Debug.Assert(rowVector.Count == destination.RowCount);
        Debug.Assert(columnVector.Count == destination.ColumnCount);

        ref var columnPtr = ref MemoryMarshal.GetReference(columnVector.AsSpan());
        ref var destinationPtr = ref MemoryMarshal.GetReference(destination.AsSpan());

        var rowCount = (nuint)rowVector.Count;
        var columnCount = (nuint)columnVector.Count;

        nuint mdSize = (nuint)SimdVector.Count;

        for (nuint row = 0; row < rowCount; row++)
        {
            var rowValue = new SimdVector(rowVector[row]);
            var rowOffset = row * columnCount;

            nuint column = 0;
            for (; column <= columnCount - mdSize; column += mdSize)
            {
                var columnValues = SimdVectorHelper.LoadUnsafe(ref columnPtr, column);
                var destinationValues = rowValue * columnValues;

                SimdVectorHelper.StoreUnsafe(destinationValues, ref destinationPtr, rowOffset + column);
            }

            for (; column < columnCount; column++)
            {
                destination[rowOffset + column] = rowVector[row] * columnVector[column];
            }
        }
    }

    public static void MultiplyToSelf(this Vector vector, Weight factor) => MultiplyTo(vector, factor, vector);
    public static Vector Multiply(this Vector vector, Weight factor)
    {
        var result = Vector.Create(vector.Count);
        MultiplyTo(vector, factor, result);
        return result;
    }
    public static void MultiplyTo(this Vector vector, Weight factor, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        TensorPrimitives.Multiply(vector.AsSpan(), factor, destination.AsSpan());
    }

    public static void DivideToSelf(this Vector vector, Weight divisor) => DivideTo(vector, divisor, vector);
    public static Vector Divide(this Vector vector, Weight divisor)
    {
        var destination = Vector.Create(vector.Count);
        DivideTo(vector, divisor, destination);
        return destination;
    }
    public static void DivideTo(this Vector vector, Weight divisor, Vector destination)
    {
        MultiplyTo(vector, 1 / divisor, destination);
    }

    public static void AddToSelf(this Vector left, Vector right) => AddTo(left, right, left);
    public static Vector Add(this Vector left, Vector right)
    {
        var result = Vector.Create(left.Count);
        AddTo(left, right, result);
        return result;
    }

    public static void AddTo(this Vector left, Vector right, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), destination.AsSpan());
    }

    public static void SubtractPointwiseToSelf(this Vector left, Weight right) => SubtractPointwiseTo(left, right, left);
    public static Vector SubtractPointwise(this Vector left, Weight right)
    {
        var destination = Vector.Create(left.Count);
        SubtractPointwiseTo(left, right, destination);
        return destination;
    }
    public static void SubtractPointwiseTo(this Vector left, Weight right, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(left, destination);
        TensorPrimitives.Subtract(left.AsSpan(), right, destination.AsSpan());
    }

    public static void SubtractToSelf(this Vector left, Vector right) => SubtractTo(left, right, left);
    public static Vector Subtract(this Vector left, Vector right)
    {
        var destination = Vector.Create(left.Count);
        SubtractTo(left, right, destination);
        return destination;
    }
    public static void SubtractTo(this Vector left, Vector right, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        TensorPrimitives.Subtract(left.AsSpan(), right.AsSpan(), destination.AsSpan());
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
    public static Weight Min(this Vector vector) => TensorPrimitives.Min<Weight>(vector.AsSpan());

    public static Vector CreateCopy(this Vector vector)
    {
        var copy = Vector.Create(vector.Count);
        vector.AsSpan().CopyTo(copy.AsSpan());
        return copy;
    }

    public static void CopyTo(this Vector vector, Vector destination)
    {
        NumericsDebug.AssertSameDimensions(vector, destination);
        vector.AsSpan().CopyTo(destination.AsSpan());
    }

    public static void ResetZero(this Vector vector) => vector.AsSpan().Clear();
}
