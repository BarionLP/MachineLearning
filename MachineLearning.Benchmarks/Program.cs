using System.Numerics.Tensors;
using BenchmarkDotNet.Running;
using MachineLearning.Benchmarks;


var Size = 8;
var random = new Random(69);
var data = Enumerable.Range(0, Size).Select(v => random.NextDouble()).ToArray();
var tensor = Tensor.Create(data, [Size]);
var vector = Vector.Of(data);

var tensor2 = Tensor.Add(tensor, tensor);

Console.WriteLine($"[{string.Join(' ', tensor.Select(d => d.ToString("F2")))}]");
Console.WriteLine($"[{string.Join(' ', tensor2.Select(d => d.ToString("F2")))}]");
Console.WriteLine(vector);

//TensorPrimitives.Add(tensor.AsReadOnlyTensorSpan(), (ReadOnlySpan<double>)vector.AsSpan(), tensor2.AsTensorSpan());

BenchmarkRunner.Run<TensorBenchmarks>();