using System.Numerics.Tensors;

namespace MachineLearning.Benchmarks;

[MemoryDiagnoser(false), ShortRunJob]
public class TensorBenchmarks
{
    internal Vector vector_l = default!;
    internal Vector vector_r = default!;
    internal Vector result_v = default!;

    [Params(255, 2048*8)]
    public int Size = 16;

    [GlobalSetup]
    public void GlobalSetup(){
        var random = new Random(69);
        var data_l = Enumerable.Range(0, Size).Select(v=> random.NextDouble()).ToArray();
        var data_r = Enumerable.Range(0, Size).Select(v=> random.NextDouble()).ToArray();
        vector_l = Vector.Of(data_l);
        vector_r = Vector.Of(data_r);
        result_v = Vector.Create(Size);
    }

    [Benchmark(Baseline = true)]
    public double MyVector() {
        return vector_l.Sum();
    }
    
    [Benchmark]
    public double Vector_Primitives() {
        return TensorPrimitives.Sum<double>(vector_l.AsSpan());
    }

    //TensorPrimitives is faster (keep my type and use AsSpan())

}
