using System.Buffers;
using Ametrin.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using ML.Core.Training;
using Weight = float;

BenchmarkRunner.Run<Benchmarks>();

[MemoryDiagnoser(false)]
public class Benchmarks
{
    [Params(512)]
    public int Size { get; set; }
    private Vector logits;
    private Vector expected;
    private Vector destination;

    private AdamOptimizer optimizer = new() { LearningRate = 0.01f };


    [GlobalSetup]
    public void Setup()
    {
        logits = Vector.Create(Size);
        logits.Uniform(-1, 1, new Random(43));
        expected = Vector.Create(Size);
        expected.Uniform(-1, 1, new Random(68));
        destination = Vector.Create(Size);
        optimizer.Init();
    }

    [Benchmark]
    public void Delegates()
    {
        // SpanOperations.MapTo(logits.AsSpan(), expected.AsSpan(), destination.AsSpan(), optimizer.WeightReduction, optimizer.WeightReduction);
    }

    [Benchmark]
    public void Static()
    {
        // SpanOperations.MapTo(optimizer.WeightReductionOperation, logits.AsSpan(), expected.AsSpan(), destination.AsSpan());
    }
}