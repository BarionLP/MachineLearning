using System.Diagnostics;
using System.Numerics.Tensors;

namespace MachineLearning.Domain.Numerics;

public interface Tensor
{
    public int RowCount { get; }
    public int ColumnCount { get; }
    public int LayerCount { get; }
    public int FlatCount { get; }
    public ref Weight this[int row, int column, int layer] { get; }
    public ref Weight this[nuint flatIndex] { get; }
    internal Vector Storage { get; }

    public Span<Weight> AsSpan();
    //public Vector RowRef(int rowIndex);

    public static Tensor CreateCube(int size) => Create(size, size, size);
    public static Tensor Create(int rowCount, int columnCount, int layerCount) => new TensorFlat(rowCount, columnCount, layerCount, Vector.Create(rowCount * columnCount * layerCount));
    public static Tensor Of(int rowCount, int columnCount, int layerCount, double[] storage) => Of(rowCount, columnCount, layerCount, Vector.Of(storage));
    public static Tensor Of(int rowCount, int columnCount, int layerCount, Vector storage)
    {
        if(storage.Count != rowCount * columnCount * layerCount)
        {
            throw new ArgumentException("storage size does not match specified dimensions");
        }

        return new TensorFlat(rowCount, columnCount, layerCount, storage);
    }

    public static Tensor OfSize(Tensor template) => Create(template.RowCount, template.ColumnCount, template.LayerCount);
}

public readonly struct TensorFlat(int rowCount, int columnCount, int layerCount, Vector storage) : Tensor
{
    public ref double this[int row, int column, int layer] => ref Storage[GetFlatIndex(row, column, layer)];
    public ref double this[nuint flatIndex] => ref Storage[flatIndex];


    public int RowCount { get; } = rowCount;
    public int ColumnCount { get; } = columnCount;
    public int LayerCount { get; } = layerCount;

    public int FlatCount => Storage.Count;

    public Vector Storage { get; } = storage;

    public Span<double> AsSpan() => Storage.AsSpan();

    internal int GetFlatIndex(int row, int column, int layer)
    {
#if DEBUG
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, RowCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, ColumnCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, LayerCount);
#endif

        return layer * RowCount * ColumnCount +  row * ColumnCount + column;
    }
}

public static class TensorHelper
{
    public static void AddToSelf(this Tensor left, Tensor right)
    {
        AssertCountEquals(left, right);
        AddUnsafe(left, right, left);
    }
    public static Tensor Add(this Tensor left, Tensor right)
    {
        AssertCountEquals(left, right);
        var destination = Tensor.OfSize(left);
        AddUnsafe(left, right, destination);
        return destination;
    }
    public static void Add(this Tensor left, Tensor right, Tensor destination)
    {
        AssertCountEquals(left, right, destination);
        AddUnsafe(left, right, destination);
    }
    private static void AddUnsafe(Tensor left, Tensor right, Tensor destination) => TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), destination.AsSpan());
    
    public static void PointwiseMultiplyToSelf(this Tensor left, Tensor right)
    {
        AssertCountEquals(left, right);
        PointwiseMultiplyUnsafe(left, right, left);
    }
    public static Tensor PointwiseMultiply(this Tensor left, Tensor right)
    {
        AssertCountEquals(left, right);
        var destination = Tensor.OfSize(left);
        PointwiseMultiplyUnsafe(left, right, destination);
        return destination;
    }
    public static void PointwiseMultiply(this Tensor left, Tensor right, Tensor destination)
    {
        AssertCountEquals(left, right, destination);
        PointwiseMultiplyUnsafe(left, right, destination);
    }
    private static void PointwiseMultiplyUnsafe(Tensor left, Tensor right, Tensor destination) => TensorPrimitives.Multiply(left.AsSpan(), right.AsSpan(), destination.AsSpan());

    public static Matrix LayerRef(this Tensor tensor, int layer) => new TensorLayerReference(layer, tensor);


    const string TENSOR_COUNT_MISMATCH = "tensors must match in size!";
    [Conditional("DEBUG")]
    private static void AssertCountEquals(Tensor a, Tensor b)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount && a.LayerCount == b.LayerCount, TENSOR_COUNT_MISMATCH);
    }
    [Conditional("DEBUG")]
    private static void AssertCountEquals(Tensor a, Tensor b, Tensor c)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount && a.LayerCount == b.LayerCount &&
                     a.ColumnCount == b.ColumnCount && b.ColumnCount == c.ColumnCount && a.LayerCount == c.LayerCount, TENSOR_COUNT_MISMATCH);
    }
}