using System.Numerics.Tensors;

namespace Ametrin.Numerics;

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
    public static Tensor Of(int rowCount, int columnCount, int layerCount, Weight[] storage) => Of(rowCount, columnCount, layerCount, Vector.Of(storage));
    public static Tensor Of(int rowCount, int columnCount, int layerCount, Vector storage)
    {
        if (storage.Count != rowCount * columnCount * layerCount)
        {
            throw new ArgumentException("storage size does not match specified dimensions");
        }

        return new TensorFlat(rowCount, columnCount, layerCount, storage);
    }

    public static Tensor OfSize(Tensor template) => Create(template.RowCount, template.ColumnCount, template.LayerCount);
}

public readonly struct TensorFlat(int rowCount, int columnCount, int layerCount, Vector storage) : Tensor
{
    public ref Weight this[int row, int column, int layer] => ref Storage[GetFlatIndex(row, column, layer)];
    public ref Weight this[nuint flatIndex] => ref Storage[flatIndex];


    public int RowCount { get; } = rowCount;
    public int ColumnCount { get; } = columnCount;
    public int LayerCount { get; } = layerCount;

    public int FlatCount => Storage.Count;

    public Vector Storage { get; } = storage;

    public Span<Weight> AsSpan() => Storage.AsSpan();

    internal int GetFlatIndex(int row, int column, int layer)
    {
#if DEBUG
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, RowCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, ColumnCount);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, LayerCount);
#endif

        return layer * RowCount * ColumnCount + row * ColumnCount + column;
    }
}

public static class TensorHelper
{
    public static void AddToSelf(this Tensor left, Tensor right)
    {
        NumericsDebug.AssertSameDimensions(left, right);
        AddUnsafe(left, right, left);
    }
    public static Tensor Add(this Tensor left, Tensor right)
    {
        NumericsDebug.AssertSameDimensions(left, right);
        var destination = Tensor.OfSize(left);
        AddUnsafe(left, right, destination);
        return destination;
    }
    public static void Add(this Tensor left, Tensor right, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        AddUnsafe(left, right, destination);
    }
    private static void AddUnsafe(Tensor left, Tensor right, Tensor destination) => TensorPrimitives.Add(left.AsSpan(), right.AsSpan(), destination.AsSpan());

    public static void PointwiseMultiplyToSelf(this Tensor left, Tensor right)
    {
        NumericsDebug.AssertSameDimensions(left, right);
        PointwiseMultiplyUnsafe(left, right, left);
    }
    public static Tensor PointwiseMultiply(this Tensor left, Tensor right)
    {
        NumericsDebug.AssertSameDimensions(left, right);
        var destination = Tensor.OfSize(left);
        PointwiseMultiplyUnsafe(left, right, destination);
        return destination;
    }
    public static void PointwiseMultiplyTo(this Tensor left, Tensor right, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(left, right, destination);
        PointwiseMultiplyUnsafe(left, right, destination);
    }
    private static void PointwiseMultiplyUnsafe(Tensor left, Tensor right, Tensor destination) => TensorPrimitives.Multiply(left.AsSpan(), right.AsSpan(), destination.AsSpan());

    public static Matrix LayerRef(this Tensor tensor, int layer) => new TensorLayerReference(layer, tensor);

    public static Tensor CreateCopy(this Tensor tensor)
    {
        var copy = Tensor.Create(tensor.RowCount, tensor.ColumnCount, tensor.LayerCount);
        tensor.CopyTo(copy);
        return copy;
    }

    public static void CopyTo(this Tensor tensor, Tensor destination)
    {
        NumericsDebug.AssertSameDimensions(tensor, destination);
        tensor.AsSpan().CopyTo(destination.AsSpan());
    }

    public static void ResetZero(this Tensor tensor) => tensor.AsSpan().Clear();
}