using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace MachineLearning.Domain.Numerics;

// must be a continuous chunk of memory for current simd to work 
public interface Matrix {
    public int RowCount { get; }
    public int ColumnCount { get; }
    public int FlatCount { get; }
    public ref Weight this[int row, int column] { get; }
    public ref Weight this[nuint flatIndex] { get; }

    public Span<Weight> AsSpan();

    public static Matrix Create(int rowCount, int columnCount) => new MatrixFlatArray(rowCount, columnCount, new Weight[rowCount*columnCount]);
    public static Matrix Of(int rowCount, int columnCount, Weight[] storage) {
        if(storage.Length / columnCount != rowCount ) {
            throw new ArgumentException("storage size does not match specified");
        }

        return new MatrixFlatArray(rowCount, columnCount, storage);
    }
}

internal readonly struct MatrixFlatArray(int rowCount, int columnCount, Weight[] storage) : Matrix {
    private readonly Weight[] _storage = storage;
    public int RowCount { get; } = rowCount;
    public int ColumnCount { get; } = columnCount;
    public int FlatCount => _storage.Length;

    public ref Weight this[int row, int column] => ref _storage[GetFlatIndex(row, column)];
    public ref Weight this[nuint flatIndex] => ref _storage[flatIndex];
    public Span<Weight> AsSpan() => _storage.AsSpan();
    //public Span<Weight> AsSpan(int flatStartIndex, int length) => _storage.AsSpan(flatStartIndex, length);

    public override string ToString() {
        var sb = new StringBuilder();
        sb.AppendLine($"Matrix ({RowCount}x{ColumnCount}):");
        for(int i = 0; i < RowCount; i++) {
            for(int j = 0; j < ColumnCount; j++) {
                sb.Append(this[i, j].ToString("F2")).Append(' ');
            }
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private int GetFlatIndex(int row, int column) => row * ColumnCount + column;
}


public static class MatrixHelper {

    public static Vector Multiply(this Matrix matrix, Vector vector) {
        var result = Vector.Create(matrix.RowCount);
        Multiply(matrix, vector, result);
        return result;
    }

    public static void Multiply(this Matrix matrix, Vector vector, Vector result) {
        MultiplySimd(matrix, vector, result);
        
        //for(int row = 0; row < matrix.RowCount; row++) {
        //    result[row] = 0;
        //    for(int column = 0; column < matrix.ColumnCount; column++) {
        //        result[row] += matrix[row, column] * vector[column];
        //    }
        //}
    }
    
    public static void MultiplySimd(this Matrix matrix, Vector vector, Vector result) {
        Debug.Assert(vector.Count == matrix.ColumnCount);
        Debug.Assert(result.Count == matrix.RowCount);

        ref var matrixPtr = ref MemoryMarshal.GetReference(matrix.AsSpan());
        ref var vectorPtr = ref MemoryMarshal.GetReference(vector.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());

        var mdSize = (nuint) SimdVector.Count;
        var rowCount = (nuint) matrix.RowCount;
        var columnCount = (nuint) matrix.ColumnCount;


        for(nuint row = 0; row < rowCount; row++) {
            nuint rowOffset = row * columnCount;
            Weight sum = 0;
            
            nuint column = 0;
            for(; column + mdSize <= columnCount; column += mdSize) {
                var matrixVec = SimdVectorHelper.LoadUnsafe(ref matrixPtr, rowOffset + column);
                var vectorVec = SimdVectorHelper.LoadUnsafe(ref vectorPtr, column);
                sum += SimdVectorHelper.Dot(matrixVec, vectorVec);
            }

            for(; column < columnCount; column++) {
                sum += matrix[row * columnCount + column] * vector[column];
            }

            result[row] = sum;
        }
    }

    public static void MapInPlace(this Matrix matrix, Func<Weight, Weight> map) => matrix.Map(map, matrix);
    public static Matrix Map(this Matrix matrix, Func<Weight, Weight> map) {
        var result = Matrix.Create(matrix.RowCount, matrix.ColumnCount);
        matrix.Map(map, result);
        return result;
    }
    public static void Map(this Matrix matrix, Func<Weight, Weight> map, Matrix result) {
        var dataSize = (nuint) matrix.FlatCount;
        for(nuint i = 0; i < dataSize; i++) {
            result[i] = map.Invoke(matrix[i]);
        }
    }

    public static void MapInPlaceOnFirst(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map) => matrices.Map(matrices.a, map);
    public static Matrix Map(this (Matrix a, Matrix b) matrices, Func<Weight, Weight, Weight> map) {
        var result = Matrix.Create(matrices.a.RowCount, matrices.a.ColumnCount);
        matrices.Map(result, map);
        return result;
    }
    public static void Map(this (Matrix a, Matrix b) matrices, Matrix result, Func<Weight, Weight, Weight> map) {
        var dataSize = (nuint) matrices.a.FlatCount;
        for(nuint i = 0; i < dataSize; i++) {
            result[i] = map.Invoke(matrices.a[i], matrices.b[i]);
        }
    }

    public static void AddInPlace(this Matrix left, Matrix right) {
        Add(left, right, left);
    }
    public static Matrix Add(this Matrix left, Matrix right) {
        var result = Matrix.Create(left.RowCount, left.ColumnCount);
        Add(left, right, result);
        return result;
    }

    //TODO: test
    public static void Add(this Matrix left, Matrix right, Matrix result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint dataSize = (nuint) left.FlatCount;

        nuint index = 0;
        for(; index + mdSize <= dataSize; index += mdSize) {
            var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
            var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
            SimdVectorHelper.StoreUnsafe(vec1 + vec2, ref resultPtr, index);
        }

        for(; index < dataSize; index++) {
            result[index] = left[index] + right[index];
        }
    }
    
    public static void SubtractInPlace(this Matrix left, Matrix right) {
        Subtract(left, right, left);
    }
    public static Matrix Subtract(this Matrix left, Matrix right) {
        var result = Matrix.Create(left.RowCount, left.ColumnCount);
        Subtract(left, right, result);
        return result;
    }

    //TODO: test
    public static void Subtract(this Matrix left, Matrix right, Matrix result) {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint dataSize = (nuint) left.FlatCount;

        nuint index = 0;
        for(; index + mdSize <= dataSize; index += mdSize) {
            var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
            var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
            SimdVectorHelper.StoreUnsafe(vec1 - vec2, ref resultPtr, index);
        }

        for(; index < dataSize; index++) {
            result[index] = left[index] - right[index];
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