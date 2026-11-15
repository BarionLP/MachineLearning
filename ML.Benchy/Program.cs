using System.Buffers;
using Ametrin.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Running;
using ML.Core.Training;
using Weight = float;

var b = new NodeGraphBenchmarks();
b.Setup();

Console.WriteLine(b.Module());
Console.WriteLine(b.Node());

// BenchmarkRunner.Run<NodeGraphBenchmarks>();

Console.WriteLine(System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription);
Console.WriteLine(Environment.Version);

BenchmarkRunner.Run<Benchmarks>();

[MemoryDiagnoser(false)]
public class Benchmarks
{
    public Vector a = Vector.Create(1024*4);
    public Vector b = Vector.Create(1024*4);

    [Benchmark]
    public void Add() => a.AddToSelf(b);
}