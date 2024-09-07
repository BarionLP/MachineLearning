using BenchmarkDotNet.Running;
using MachineLearning.Benchmarks;


//var tests = new TensorBenchmarks();
//tests.GlobalSetup();

//Console.WriteLine(tests.vector_l);
//Console.WriteLine(tests.vector_r);

//Console.WriteLine("Results:");
//tests.MyVector();
//Console.WriteLine(tests.result_v);
//tests.Vector_Primitives();
//Console.WriteLine(tests.result_v);

var test = new RandomBenchmarks();
test.Setup();

test.Multiply_Old();
Console.WriteLine(test.result);
test.Multiply_New();
Console.WriteLine(test.result);

BenchmarkRunner.Run<RandomBenchmarks>();