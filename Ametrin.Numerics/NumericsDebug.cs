using System.Diagnostics;

namespace Ametrin.Numerics;

public static class NumericsDebug
{
    const string VECTOR_SIZE_MISMATCH = "Vectors must match in size";
    const string MATRIX_DIMENSION_MISMATCH = "Matrices must match in dimensions";
    const string TENSOR_DIMENSION_MISMATCH = "Tensors must match in dimensions";

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Matrix a, Matrix b)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount, MATRIX_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Matrix a, Matrix b, Matrix c)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount &&
                     a.RowCount == c.RowCount && a.ColumnCount == c.ColumnCount,
                     MATRIX_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Matrix a, Matrix b, Matrix c, Matrix d)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount &&
                     a.RowCount == c.RowCount && a.ColumnCount == c.ColumnCount &&
                     a.RowCount == d.RowCount && a.ColumnCount == d.ColumnCount,
                     MATRIX_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Vector a, Vector b)
    {
        Debug.Assert(a.Count == b.Count, VECTOR_SIZE_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Vector a, Vector b, Vector c)
    {
        Debug.Assert(a.Count == b.Count &&
                     a.Count == c.Count,
                     VECTOR_SIZE_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Vector a, Vector b, Vector c, Vector d)
    {
        Debug.Assert(a.Count == b.Count &&
                     a.Count == c.Count &&
                     a.Count == d.Count,
                     VECTOR_SIZE_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Tensor a, Tensor b)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount && a.LayerCount == b.LayerCount, TENSOR_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertSameDimensions(Tensor a, Tensor b, Tensor c)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount && a.LayerCount == b.LayerCount &&
                     a.RowCount == c.RowCount && a.ColumnCount == c.ColumnCount && a.LayerCount == c.LayerCount,
                     TENSOR_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG")]
    public static void AssertValidNumbers(Vector vector)
    {
        Debug.Assert(!vector.AsSpan().Contains(Weight.NaN), "Vector contains invalid numbers");
    }

    [Conditional("DEBUG")]
    public static void AssertValidNumbers(Matrix vector)
    {
        Debug.Assert(!vector.AsSpan().Contains(Weight.NaN), "Matrix contains invalid numbers");
    }
}
