using System.Diagnostics;
using System.Runtime.InteropServices;
using SimdVector = System.Numerics.Vector<double>;
using SimdVectorHelper = System.Numerics.Vector;

namespace MachineLearning.Domain.Numerics;

// must be a continuous chunk of memory for current simd to work 
public interface Matrix {
    public int RowCount { get; }
    public int ColumnCount { get; }
    public int FlatCount { get; }
    public ref Weight this[int row, int column] { get; }
    public ref Weight this[int flatIndex] { get; }

    public Span<Weight> AsSpan();

    public static Matrix Create(int rowCount, int columnCount) => new MatrixFlatArray(rowCount, columnCount, new Weight[rowCount*columnCount]);
    public static Matrix Of(int rowCount, int columnCount, Weight[] storage) {
        if(storage.Length / columnCount != rowCount ) {
            throw new ArgumentException("storage size does not match specified");
        }

        return new MatrixFlatArray(rowCount, columnCount, new Weight[rowCount * columnCount]);
    }
}

internal readonly struct MatrixFlatArray(int rowCount, int columnCount, Weight[] storage) : Matrix {
    private readonly Weight[] _storage = storage;
    public int RowCount { get; } = rowCount;
    public int ColumnCount { get; } = columnCount;
    public int FlatCount => _storage.Length;

    public ref Weight this[int row, int column] => ref this[GetFlatIndex(row, column)];
    public ref Weight this[int flatIndex] => ref _storage[flatIndex];
    public Span<Weight> AsSpan() => _storage.AsSpan();
    //public Span<Weight> AsSpan(int flatStartIndex, int length) => _storage.AsSpan(flatStartIndex, length);

    private int GetFlatIndex(int row, int column) => row * ColumnCount + column;
}


public static class MatrixHelper {

    public static Vector Multiply(this Matrix matrix, Vector vector) {
        var result = Vector.Create(matrix.RowCount);
        matrix.Multiply(vector, result);
        return result;
    }
    // TODO: simd and test
    public static void Multiply(this Matrix matrix, Vector vector, Vector result) {
        Debug.Assert(vector.Count == matrix.ColumnCount);
        Debug.Assert(result.Count == matrix.RowCount);
        
        for(int row = 0; row < matrix.RowCount; row++) {
            result[row] = 0;
            for(int column = 0; column < matrix.ColumnCount; column++) {
                result[row] += matrix[row, column] * vector[column];
            }
        }
    }

    public static void MapInPlace(this Matrix left, Func<Weight, Weight> map) => left.Map(map, left);
    public static Matrix Map(this Matrix left, Func<Weight, Weight> fallbackAction) {
        var result = Matrix.Create(left.RowCount, left.ColumnCount);
        left.Map(fallbackAction, result);
        return result;
    }
    public static void Map(this Matrix left, Func<Weight, Weight> fallbackAction, Matrix result) {
        for(int i = 0; i < left.FlatCount; i++) {
            result[i] = fallbackAction.Invoke(left[i]);
        }
    }

    public static void MapInPlaceOnFirst(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map) => matrices.Map(matrices.a, map);
    public static Matrix Map(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map) {
        var result = Matrix.Create(matrices.a.RowCount, matrices.a.ColumnCount);
        matrices.Map(result, map);
        return result;
    }
    public static void Map(this (Matrix a, Matrix b) matrices, Matrix result, Func<Weight, Weight, Weight> map) {
        for(int i = 0; i < matrices.a.FlatCount; i++) {
            result[i] = map.Invoke(matrices.a[i], matrices.b[i]);
        }
    }

    public static void AddInPlace(this Matrix left, Matrix right) {
        DoAdd(left, right, left);
    }
    public static Matrix Add(this Matrix left, Matrix right) {
        var target = Matrix.Create(left.RowCount, left.ColumnCount);
        DoAdd(left, right, target);
        return target;
    }
    public static void Add(this Matrix left, Matrix right, Matrix target) {
        DoAdd(left, right, target);
    }

    //TODO: test
    private static void DoAdd(this Matrix left, Matrix right, Matrix result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) left.FlatCount;

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
    
    public static void SubtractInPlace(this Matrix left, Matrix right) {
        DoSubtract(left, right, left);
    }
    public static Matrix Subtract(this Matrix left, Matrix right) {
        var target = Matrix.Create(left.RowCount, left.ColumnCount);
        DoSubtract(left, right, target);
        return target;
    }
    public static void Subtract(this Matrix left, Matrix right, Matrix target) {
        DoSubtract(left, right, target);
    }

    //TODO: test
    private static void DoSubtract(this Matrix left, Matrix right, Matrix result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) left.FlatCount;

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
    public static Matrix Copy(this Matrix matrix) {
        var copy = Matrix.Create(matrix.RowCount, matrix.ColumnCount);
        matrix.AsSpan().CopyTo(copy.AsSpan());
        return copy;
    }

    //TODO: test
    public static void ResetZero(this Matrix matrix) {
        matrix.AsSpan().Clear();
    }

    private static void ThrowIfSizeMismatch(Matrix a, Matrix b) {
        if(a.RowCount != b.RowCount
        || a.ColumnCount != b.ColumnCount) {
            throw new ArgumentException("Matrices have to match in Size");
        }
    }
    private static void ThrowIfSizeMismatch(Matrix a, Matrix b, Matrix c) {
        if(a.RowCount != b.RowCount || b.RowCount != c.RowCount
        || a.ColumnCount != b.ColumnCount || b.ColumnCount != c.ColumnCount) {
            throw new ArgumentException("Matrices have to match in Size");
        }
    }
}