using System.Diagnostics;

namespace Ametrin.Numerics;

public static class SpanOperations
{
    public static void MapTo<T>(ReadOnlySpan<T> values, Span<T> destination, Func<T, T> map)
    {
        Debug.Assert(values.Length == destination.Length);
        for (int i = 0; i < values.Length; i++)
        {
            destination[i] = map(values[i]);
        }
    }

    public static void MapTo<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> destination, Func<T, T, T> map)
    {
        Debug.Assert(left.Length == right.Length && left.Length == destination.Length);

        for (int i = 0; i < left.Length; i++)
        {
            destination[i] = map(left[i], right[i]);
        }
    }
    public static void MapTo<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, ReadOnlySpan<T> c, Span<T> destination, Func<T, T, T, T> map)
    {
        Debug.Assert(a.Length == b.Length && a.Length == c.Length && a.Length == destination.Length);

        for (int i = 0; i < a.Length; i++)
        {
            destination[i] = map(a[i], b[i], c[i]);
        }
    }
}
