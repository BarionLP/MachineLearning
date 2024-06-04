using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace MachineLearning.Domain.Numerics;

public interface Vector {
    public static readonly Vector Empty = new VectorSimple(0, []);
    public int Count { get; }
    public ref Weight this[int index] { get; }
    public ref Weight this[nuint index] { get; }
    public Span<Weight> this[int index, int count] { get; }
    public Span<Weight> AsSpan();

    public static Vector Create(int count) => new VectorSimple(count, new Weight[count]);
    public static Vector Of(Weight[] array) => new VectorSimple(array.Length, array);
}

internal readonly struct VectorSimple(int count, Weight[] storage) : Vector {
    private readonly Weight[] _storage = storage;
    public int Count { get; } = count;
    public ref Weight this[int index] => ref _storage[index];
    public ref Weight this[nuint index] => ref _storage[index];
    public Span<Weight> this[int index, int count] => new(_storage, index, count); // has built-in bound checks
    public Span<Weight> AsSpan() => new(_storage, 0, Count);
    public override string ToString() => $"[{string.Join(' ', _storage.Select(d=> d.ToString("F2")))}]";
}

public static class VectorHelper {
    public static Weight Sum(this Vector vector) {
        ref var ptr = ref MemoryMarshal.GetReference(vector.AsSpan());
        nuint length = (nuint) vector.Count;

        var accumulator = SimdVector.Zero;
        nuint index = 0;

        var limit = length - (nuint) SimdVector.Count;
        for(; index < limit; index += (nuint) SimdVector.Count) {
            accumulator += SimdVectorHelper.LoadUnsafe(ref ptr, index);
        }

        var result = SimdVectorHelper.Sum(accumulator);

        for(; index < length; index++) {
            result += Unsafe.Add(ref ptr, index);
        }

        return result;
    }

    public static void MapInPlace(this Vector vector, Func<Weight, Weight> map) => vector.Map(map, vector);
    public static Vector Map(this Vector vector, Func<Weight, Weight> map) {
        var result = Vector.Create(vector.Count);
        vector.Map(map, result);
        return result;
    }
    public static void Map(this Vector vector, Func<Weight, Weight> map, Vector result) {
        for(int i = 0; i < vector.Count; i++) {
            result[i] = map.Invoke(vector[i]);
        }
    }
    
    public static void MapInPlace(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap) => vector.Map(simdMap, fallbackMap, vector);
    public static void Map(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap, Vector result) {
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        var length = (nuint) vector.Count;

        nuint index = 0;
        for(; index + mdSize <= length; index += mdSize) {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe(simdMap.Invoke(simdVector), ref resultPtr, index);
        }

        for(; index < length; index++) {
            result[index] = fallbackMap.Invoke(vector[index]);
        }
    }

    public static void MapInFirst(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map) => vectors.Map(map, vectors.a);
    public static Vector Map(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map) {
        var result = Vector.Create(vectors.a.Count);
        vectors.Map(map, result);
        return result;
    }
    public static void Map(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map, Vector result) {
        for(int i = 0; i < vectors.a.Count; i++) {
            result[i] = map.Invoke(vectors.a[i], vectors.b[i]);
        }
    }

    public static void PointwiseMultiplyInPlace(this Vector left, Vector right) => left.PointwiseMultiply(right, left);
    public static Vector PointwiseMultiply(this Vector left, Vector right) {
        var result = Vector.Create(left.Count);
        left.PointwiseMultiply(right, result);
        return result;
    }
    public static void PointwiseMultiply(this Vector left, Vector right, Vector result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        var length = (nuint) left.Count;

        nuint index = 0;
        for(; index + mdSize <= length; index += mdSize) {
            var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
            var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
            SimdVectorHelper.StoreUnsafe(vec1 * vec2, ref resultPtr, index);
        }

        for(; index < length; index++) {
            result[index] = left[index] * right[index];
        }
    }

    public static Vector Multiply(this Vector vector, Matrix matrix) {
        var result = Vector.Create(matrix.ColumnCount);
        vector.Multiply(matrix, result);
        return result;
    }
    // cannot be simded because row entries in matrix are not next to each other in memory
    public static void Multiply(this Vector vector, Matrix matrix, Vector result) {
        Debug.Assert(vector.Count == matrix.RowCount);
        Debug.Assert(result.Count == matrix.ColumnCount);

        for(int column = 0; column < matrix.ColumnCount; column++) {
            result[column] = 0;
            for(int row = 0; row < matrix.RowCount; row++) {
                result[column] += vector[row] * matrix[row, column];
            }
        }
    }


    public static Matrix MultiplyToMatrix(Vector rowVector, Vector columnVector) { 
        var result = Matrix.Create(rowVector.Count, columnVector.Count);
        MultiplyToMatrix(rowVector, columnVector, result);
        return result;
    }

    // TODO: simd and test
    public static void MultiplyToMatrix(Vector rowVector, Vector columnVector, Matrix result) {
        
        MultiplyToMatrixSimd(rowVector, columnVector, result);
        //for(int row = 0; row < rowVector.Count; row++) {
        //    for(int column = 0; column < columnVector.Count; column++) {
        //        result[row, column] = rowVector[row] * columnVector[column];
        //    }
        //}
    }

    public static void MultiplyToMatrixSimd(Vector rowVector, Vector columnVector, Matrix result) {
        int rowCount = rowVector.Count;
        int colCount = columnVector.Count;

        if(rowCount == 0 || colCount == 0 || rowCount * colCount != result.FlatCount)
            throw new ArgumentException("Invalid dimensions for the vectors or result matrix.");

        int simdLength = SimdVector.Count;

        for(int row = 0; row < rowCount; row++) {
            var rowValue = new SimdVector(rowVector[row]);

            int col;
            for(col = 0; col <= colCount - simdLength; col += simdLength) {
                var colValues = new SimdVector(columnVector.AsSpan().Slice(col, SimdVector.Count));
                var resultValues = rowValue * colValues;

                for(int i = 0; i < simdLength; i++) {
                    result[row, col+i] = resultValues[i];
                }
            }

            for(; col < colCount; col++) {
                result[row, col] = rowVector[row] * columnVector[col];
            }
        }
    }


    public static void MultiplyInPlace(this Vector vector, Weight factor) => Multiply(vector, factor, vector);
    public static Vector Multiply(this Vector vector, Weight factor) {
        var result = Vector.Create(vector.Count);
        Multiply(vector, factor, result);
        return result;
    }
    public static void Multiply(this Vector vector, Weight factor, Vector result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) vector.Count;

        nuint index = 0;
        for(; index + mdSize <= length; index += mdSize) {
            var md = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
            SimdVectorHelper.StoreUnsafe(md * factor, ref resultPtr, index);
        }

        // TODO: does unsafe pointer math helps significantly? (see sum method) improve in other methods too
        for(; index < length; index++) {
            result[index] = vector[index] * factor;
        }
    }
    
    public static void DivideInPlace(this Vector vector, Weight divisor) => Divide(vector, divisor, vector);
    public static Vector Divide(this Vector vector, Weight divisor) {
        var result = Vector.Create(vector.Count);
        Divide(vector, divisor, result);
        return result;
    }
    public static void Divide(this Vector vector, Weight divisor, Vector result) {
        Multiply(vector, 1/divisor, result);
    }
    
    public static void AddInPlace(this Vector left, Vector right) => Add(left, right, left);
    public static Vector Add(this Vector left, Vector right) {
        var result = Vector.Create(left.Count);
        Add(left, right, result);
        return result;
    }
    public static void Add(this Vector left, Vector right, Vector result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) left.Count;

        nuint index = 0;
        for(; index + mdSize <= length; index += mdSize) {
            var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
            var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
            SimdVectorHelper.StoreUnsafe(vec1 + vec2, ref resultPtr, index);
        }

        for(; index < length; index++) {
            result[index] = left[index] + right[index];
        }
    }
    
    public static void SubtractInPlace(this Vector left, Vector right) => Subtract(left, right, left);
    public static Vector Subtract(this Vector left, Vector right) {
        var result = Vector.Create(left.Count);
        Subtract(left, right, result);
        return result;
    }
    public static void Subtract(this Vector left, Vector right, Vector result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) left.Count;

        nuint index = 0;
        for(; index + mdSize <= length; index += mdSize) {
            var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
            var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
            SimdVectorHelper.StoreUnsafe(vec1 - vec2, ref resultPtr, index);
        }

        for(; index < length; index++) {
            result[index] = left[index] - right[index];
        }
    }

    //TODO: test
    public static int MaximumIndex(this Vector vector) {
        var maxIndex = 0;
        for(int i = 1; i < vector.Count; i++) {
            if(vector[i] > vector[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    //TODO: test
    public static Vector Copy(this Vector vector) {
        var copy = Vector.Create(vector.Count);
        vector.AsSpan().CopyTo(copy.AsSpan());
        return copy;
    }

    //TODO: test
    public static void ResetZero(this Vector vector) {
        vector.AsSpan().Clear();
    }

    private static void ThrowIfSizeMismatch(Vector a, Vector b) {
        if(a.Count != b.Count) {
            throw new ArgumentException("Vectors have to match in Size");
        }
    }
    private static void ThrowIfSizeMismatch(Vector a, Vector b, Vector c) {
        if(a.Count != b.Count || b.Count != c.Count) {
            throw new ArgumentException("Vectors have to match in Size");
        }
    }
}
