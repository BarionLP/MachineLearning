using System.Buffers;
using Ametrin.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Weight = float;

// BenchmarkRunner.Run<Benchmarks>();

[MemoryDiagnoser(false)]
public class Benchmarks
{
    [Params(10, 512)]
    public int Size { get; set; }
    private Vector logits;
    private Vector expected;
    // private Vector destination;

    [GlobalSetup]
    public void Setup()
    {
        logits = Vector.Create(Size);
        logits.Uniform(-1, 1, new Random(43));
        expected = Vector.Create(Size);
        expected.Uniform(-1, 1, new Random(68));
        // destination = Vector.Create(Size);
    }
}