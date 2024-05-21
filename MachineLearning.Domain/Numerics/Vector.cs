using System.Collections;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SimdVector = System.Numerics.Vector<double>;
using SimdVectorHelper = System.Numerics.Vector;

namespace MachineLearning.Domain.Numerics;

public interface Vector : IEnumerable<Weight> { //TODO replace Enumerable uses, remove implementations
    public static readonly Vector Empty = new VectorSimple(0, []);
    public int Count { get; }
    public Span<Weight> this[int index, int count] { get; }
    public ref Weight this[int index] { get; }
    public Span<Weight> AsSpan();

    public static Vector Create(int count) => new VectorSimple(count, new Weight[count]);
    public static Vector Of(Weight[] array) => new VectorSimple(array.Length, array);
}

internal readonly struct VectorSimple(int count, Weight[] storage) : Vector {
    private readonly Weight[] _storage = storage;
    public int Count { get; } = count;
    public ref Weight this[int index] => ref _storage[index];
    public Span<Weight> this[int index, int count] => new(_storage, index, count); // has built-in bound checks
    public Span<Weight> AsSpan() => new(_storage, 0, Count);
    public override string ToString() => $"[{string.Join(' ', _storage.Select(d=> d.ToString("F2")))}]";

    public IEnumerator<Weight> GetEnumerator() {
        foreach(var e in _storage) {
            yield return e;
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => _storage.GetEnumerator();
}

public static class VectorHelper {
    public static Weight Sum(this Vector vector) {
        var span = vector.AsSpan();
        ref var ptr = ref MemoryMarshal.GetReference(span);
        nuint length = (nuint) span.Length;

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

    public static void MapInPlace(this Vector vector, Func<Weight, Weight> map) => vector.Map(vector, map);
    public static Vector Map(this Vector vector, Func<Weight, Weight> map) {
        var result = Vector.Create(vector.Count);
        vector.Map(result, map);
        return result;
    }
    public static void Map(this Vector vector, Vector result, Func<Weight, Weight> map) {
        for(int i = 0; i < vector.Count; i++) {
            result[i] = map.Invoke(vector[i]);
        }
    }
    
    public static void MapInPlace(this Vector vector, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap) => vector.Map(vector, simdMap, fallbackMap);
    public static void Map(this Vector vector, Vector result, Func<SimdVector, SimdVector> simdMap, Func<Weight, Weight> fallbackMap) {
        var dataCount = SimdVector.Count;
        int i = 0;

        for(; i <= vector.Count - dataCount; i += dataCount) {
            simdMap.Invoke(new SimdVector(vector[i, dataCount])).CopyTo(result[i, dataCount]);
        }

        for(; i < vector.Count; i++) {
            result[i] = fallbackMap.Invoke(vector[i]);
        }
    }

    public static void MapInPlaceOnFirst(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map) => vectors.Map(vectors.a, map);
    public static Vector Map(this (Vector a, Vector b) vectors, Func<Weight, Weight, Weight> map) {
        var result = Vector.Create(vectors.a.Count);
        vectors.Map(result, map);
        return result;
    }
    public static void Map(this (Vector a, Vector b) vectors, Vector result, Func<Weight, Weight, Weight> map) {
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
        nuint length = (nuint) left.Count;

        nuint index = 0;
        if(length > mdSize) {
            for(; index <= length - mdSize; index += mdSize) {
                var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
                var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
                SimdVectorHelper.StoreUnsafe(vec1 * vec2, ref resultPtr, index);
            }
        }

        var remaining = (int) (length - index);
        for(int i = 0; i < remaining; i++) {
            result[i] = left[i] * right[i];
        }
    }

    public static Vector Multiply(this Vector vector, Matrix matrix) {
        var result = Vector.Create(matrix.ColumnCount);
        vector.Multiply(matrix, result);
        return result;
    }
    // TODO: simd and test
    public static Vector Multiply(this Vector vector, Matrix matrix, Vector result) {
        Debug.Assert(vector.Count == matrix.RowCount);
        Debug.Assert(result.Count == matrix.ColumnCount);

        for(int column = 0; column < matrix.ColumnCount; column++) {
            result[column] = 0;
            for(int row = 0; row < matrix.RowCount; row++) {
                result[column] += vector[row] * matrix[row, column];
            }
        }

        return result;
    }

    public static Matrix MultiplyToMatrix(Vector rowVector, Vector columnVector) { 
        var result = Matrix.Create(rowVector.Count, columnVector.Count);
        MultiplyToMatrix(rowVector, columnVector, result);
        return result;
    }

    // TODO: simd and test
    public static void MultiplyToMatrix(Vector rowVector, Vector columnVector, Matrix result) {
        for(int row = 0; row < rowVector.Count; row++) {
            for(int column = 0; column < columnVector.Count; column++) {
                result[row, column] = rowVector[row] * columnVector[column];
            }
        }
    }


    public static void AddInPlace(this Vector left, Vector right) => left.Add(right, left);
    public static Vector Add(this Vector left, Vector right) {
        var result = Vector.Create(left.Count);
        left.Add(right, result);
        return result;
    }
    public static void Add(this Vector left, Vector right, Vector result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) left.Count;

        nuint index = 0;
        if(length > mdSize) {
            for(; index <= length - mdSize; index += mdSize) {
                var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
                var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
                SimdVectorHelper.StoreUnsafe(vec1 + vec2, ref resultPtr, index);
            }
        }

        var remaining = (int) (length - index);
        for(int i = 0; i < remaining; i++) {
            result[i] = left[i] + right[i];
        }
    }
    
    public static void SubtractInPlace(this Vector left, Vector right) => left.Subtract(right, left);
    public static Vector Subtract(this Vector left, Vector right) {
        var result = Vector.Create(left.Count);
        left.Subtract(right, result);
        return result;
    }
    public static void Subtract(this Vector left, Vector right, Vector result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) left.Count;

        nuint index = 0;
        if(length > mdSize) {
            for(; index <= length - mdSize; index += mdSize) {
                var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
                var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
                SimdVectorHelper.StoreUnsafe(vec1 - vec2, ref resultPtr, index);
            }
        }

        var remaining = (int) (length - index);
        for(int i = 0; i < remaining; i++) {
            result[i] = left[i] - right[i];
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