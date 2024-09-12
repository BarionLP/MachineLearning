using BenchmarkDotNet.Running;
using MachineLearning.Benchmarks;

var test = new MatrixOperationsBenchmark();
test.Setup();

test.MultiplyRowwise();
Console.WriteLine(test.resultMatrix);
test.Multiply();
Console.WriteLine(test.resultMatrix);

BenchmarkRunner.Run<RandomBenchmarks>();