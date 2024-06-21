using System.Numerics.Tensors;

namespace MachineLearning.Benchmarks;

[MemoryDiagnoser(false)]
public class TensorBenchmarks
{
    private Tensor<double> tensor = default!;
    private Vector vector = default!;
    private Vector result_v = default!;
    private Tensor<double> result_t = default!;

    [Params(255, 2048*8)]
    public int Size;

    [GlobalSetup]
    public void GlobalSetup(){
        var random = new Random(69);
        var data = Enumerable.Range(0, Size).Select(v=> random.NextDouble()).ToArray();
        tensor = Tensor.Create(data, [Size]);
        result_t = Tensor.Create<double>([Size]);
        vector = Vector.Of(data);
        result_v = Vector.Create(Size);
    }

    [Benchmark(Baseline = true)]
    public void Vector_Add() {
        vector.Add(vector, result_v);
    }

    [Benchmark]
    public void Tensor_Add() {
        var inSpan = MemoryMarshal.CreateReadOnlySpan(ref tensor[(ReadOnlySpan<nint>)[0]], (int)tensor.FlattenedLength);
        var outSpan = MemoryMarshal.CreateSpan(ref result_t[(ReadOnlySpan<nint>)[0]], (int)result_t.FlattenedLength);
        TensorPrimitives.Add(inSpan, inSpan, outSpan);
    }
    
    [Benchmark]
    public void Vector_Primitives_Add() {
        TensorPrimitives.Add(vector.AsSpan(), vector.AsSpan(), result_v.AsSpan());
    }

    //TensorPrimitives is faster (keep my type and use AsSpan())

}
