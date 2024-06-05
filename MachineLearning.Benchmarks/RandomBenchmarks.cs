using BenchmarkDotNet.Attributes;
using System.Runtime.InteropServices;
using SimdVector = System.Numerics.Vector<double>;
using SimdVectorHelper = System.Numerics.Vector;

namespace Simple.Benchy;

public class RandomBenchmarks
{

    //private double[] array = default!;
    private Vector left = default!;
    private Vector right = default!;
    private Vector result = default!;

    [Params(1024)]
    public int Count;

    [GlobalSetup]
    public void Setup()
    {
        left = Vector.Create(Count);
        right = Vector.Create(Count);
        result = Vector.Create(Count);

        for(int i = 0; i < Count; i++)
        {
            left[i] = Random.Shared.NextDouble();
            right[i] = Random.Shared.NextDouble();
        }
    }

    [Benchmark]
    public void Add_New()
    {
        ref var leftPtr = ref MemoryMarshal.GetReference(left.AsSpan());
        ref var rightPtr = ref MemoryMarshal.GetReference(right.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        nuint length = (nuint) left.Count;

        nuint index = 0;
        if(length > mdSize)
        {
            for(; index <= length - mdSize; index += mdSize)
            {
                var vec1 = SimdVectorHelper.LoadUnsafe(ref leftPtr, index);
                var vec2 = SimdVectorHelper.LoadUnsafe(ref rightPtr, index);
                SimdVectorHelper.StoreUnsafe(vec1 + vec2, ref resultPtr, index);
            }
        }

        var remaining = (int) (length - index);
        for(int i = 0; i < remaining; i++)
        {
            result[i] = left[i] + right[i];
        }
    }

    [Benchmark]
    public void Add_Old()
    {
        var dataCount = SimdVector.Count;
        var i = 0;

        for(; i <= left.Count - dataCount; i += dataCount)
        {
            var vec1 = new SimdVector(left[i, dataCount]);
            var vec2 = new SimdVector(right[i, dataCount]);
            (vec1 + vec2).CopyTo(result[i, dataCount]);
        }

        for(; i < left.Count; i++)
        {
            result[i] = left[i] + right[i];
        }
    }

    [Benchmark]
    public void Add_Loop()
    {
        for(int i = 0; i < result.Count; i++)
        {
            result[i] = left[i] + right[i];
        }
    }
}
