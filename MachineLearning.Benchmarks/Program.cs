using BenchmarkDotNet.Running;
using MachineLearning.Benchmarks;

var tests = new TensorBenchmarks();
tests.GlobalSetup();

Console.WriteLine(tests.vector_l);
Console.WriteLine(tests.vector_r);

Console.WriteLine("Results:");
tests.MyVector();
Console.WriteLine(tests.result_v);
tests.Vector_Primitives();
Console.WriteLine(tests.result_v);

tests.Vector_Primitives();

BenchmarkRunner.Run<TensorBenchmarks>();