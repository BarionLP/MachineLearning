namespace MachineLearning.Benchmarks;

public class ArrayAdditionBenchmarks
{
    private double[] array1 = [];
    private double[] array2 = [];

    [GlobalSetup]
    public void Setup()
    {
        int size = 1024;
        array1 = new double[size];
        array2 = new double[size];
        for(int i = 0; i < size; i++)
        {
            array1[i] = i;
            array2[i] = size - i;
        }
    }

    [Benchmark(Baseline = true)]
    public double[] Addition_Loop()
    {
        double[] result = new double[array1.Length];
        for(int i = 0; i < array1.Length; i++)
        {
            result[i] = array1[i] + array2[i];
        }
        return result;
    }

    [Benchmark]
    public double[] Addition_SIMD()
    {

        double[] result = new double[array1.Length];
        int simdLength = SimdVector.Count;
        int i = 0;

        for(; i <= array1.Length - simdLength; i += simdLength)
        {
            var vec1 = new SimdVector(array1, i);
            var vec2 = new SimdVector(array2, i);
            (vec1 + vec2).CopyTo(result, i);
        }

        for(; i < array1.Length; i++)
        {
            result[i] = array1[i] + array2[i];
        }

        return result;
    }
}
