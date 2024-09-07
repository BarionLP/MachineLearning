using System.Diagnostics;

namespace MachineLearning.Domain.Numerics;

public static class SpanOperations
{
    public static void Map<T>(ReadOnlySpan<T> values, Span<T> destination, Func<T, T> map)
    {
        Debug.Assert(values.Length == destination.Length);
        for(int i = 0; i < values.Length; i++)
        {
            destination[i] = map(values[i]);
        }
    }

    public static void Map<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> destination, Func<T, T, T> map)
    {
        Debug.Assert(left.Length == right.Length && left.Length == destination.Length);

        for(int i = 0; i < left.Length; i++)
        {
            destination[i] = map(left[i], right[i]);
        }
    }
}
