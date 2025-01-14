using System.Diagnostics;

namespace Ametrin.Numerics;

public static class NumericsDebug
{
    const string VECTOR_SIZE_MISMATCH = "Vectors don't match in size";
    const string MATRIX_DIMENSION_MISMATCH = "Matrices don't match in size";
    const string TENSOR_DIMENSION_MISMATCH = "Tensors don't match in size";

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Matrix a, Matrix b)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount, MATRIX_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Matrix a, Matrix b, Matrix c)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount &&
                     a.RowCount == c.RowCount && a.ColumnCount == c.ColumnCount,
                     MATRIX_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Matrix a, Matrix b, Matrix c, Matrix d)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount &&
                     a.RowCount == c.RowCount && a.ColumnCount == c.ColumnCount &&
                     a.RowCount == d.RowCount && a.ColumnCount == d.ColumnCount,
                     MATRIX_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Vector a, Vector b)
    {
        Debug.Assert(a.Count == b.Count, VECTOR_SIZE_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Vector a, Vector b, Vector c)
    {
        Debug.Assert(a.Count == b.Count &&
                     a.Count == c.Count,
                     VECTOR_SIZE_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Vector a, Vector b, Vector c, Vector d)
    {
        Debug.Assert(a.Count == b.Count &&
                     a.Count == c.Count &&
                     a.Count == d.Count,
                     VECTOR_SIZE_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Tensor a, Tensor b)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount && a.LayerCount == b.LayerCount, TENSOR_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertSameDimensions(Tensor a, Tensor b, Tensor c)
    {
        Debug.Assert(a.RowCount == b.RowCount && a.ColumnCount == b.ColumnCount && a.LayerCount == b.LayerCount &&
                     a.RowCount == c.RowCount && a.ColumnCount == c.ColumnCount && a.LayerCount == c.LayerCount,
                     TENSOR_DIMENSION_MISMATCH);
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertValidNumbers(Vector vector)
    {
        AssertValidNumbers(vector.AsSpan(), "Vector contains invalid numbers");
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertValidNumbers(Matrix matrix)
    {
        AssertValidNumbers(matrix.AsSpan(), "Matrix contains invalid numbers");
    }

    [Conditional("DEBUG"), StackTraceHidden]
    public static void AssertValidNumbers(ReadOnlySpan<Weight> span, string message = "Span contains invalid numbers")
    {
        Debug.Assert(!span.ContainsAny([Weight.NaN, Weight.NegativeInfinity, Weight.PositiveInfinity]), message);
    }

    [StackTraceHidden]
    public static void RequireValidNumbers(Vector vector)
    {
        RequireValidNumbers(vector.AsSpan(), "Vector contains invalid numbers");
    }
    [StackTraceHidden]
    public static void RequireValidNumbers(Matrix matrix)
    {
        RequireValidNumbers(matrix.AsSpan(), "Matrix contains invalid numbers");
    }
    [StackTraceHidden]
    public static void RequireValidNumbers(ReadOnlySpan<Weight> span, string message = "Span contains invalid numbers")
    {
        ThrowIf(span.ContainsAny([Weight.NaN, Weight.NegativeInfinity, Weight.PositiveInfinity]), message);
    }

    [StackTraceHidden]
    private static void ThrowIf(bool condition, string message)
    {
        if (condition)
        {
            throw new ArgumentException(message);
        }
    }
}
