using System.Buffers;
using Ametrin.Numerics;
using Ametrin.Numerics.Operations;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using ML.Core.Modules;
using ML.Core.Modules.Activations;
using ML.Core.Training;
using Weight = float;

var b = new Benchmarks();
b.Setup();

Console.WriteLine(b.Module());
Console.WriteLine(b.Node());

BenchmarkRunner.Run<Benchmarks>();

[MemoryDiagnoser(false)]
public class Benchmarks
{
    [Params(512)]
    public int Size { get; set; } = 12;
    private Vector input;
    private Matrix weights;
    private Vector biases;

    private IHiddenModule<Vector> module = default!;
    private IModuleSnapshot snapshot = default!;
    private IModuleGradients gradients = default!;
    private InputNode<Vector> inputNode = default!;
    private IOperationNode<Vector> node = default!;

    [GlobalSetup]
    public void Setup()
    {
        input = Vector.Create(Size);
        input.Uniform(-1, 1, new Random(43));
        weights = Matrix.CreateSquare(Size);
        weights.Uniform(-1, 1, new Random(68));
        biases = Vector.Create(Size);
        biases.Uniform(-1, 1, new Random(68));

        module = new SequenceModule<Vector>
        {
            Inner = [
                new LinearVectorModule(weights, biases),
                LeakyReLUActivation.Instance,
            ]
        };
        snapshot = module.CreateSnapshot();
        gradients = module.CreateGradients();

        inputNode = new InputNode<Vector>();
        node = new UnaryOperationNode<LeakyReLUOperation, float, Vector>
        {
            State = 0.01f,
            Source = new BinaryOperationNode<AddTensorsOperator<Vector>, Ametrin.Numerics.Operations.Empty, Vector, Vector, Vector>
            {
                State = default,
                LeftSource = new BinaryOperationNode<MatrixVectorMultiplyOperation, Ametrin.Numerics.Operations.Empty, Matrix, Vector, Vector>
                {
                    State = default,
                    LeftSource = new LearnableWeightsNode<Matrix>(weights),
                    RightSource = inputNode,
                },
                RightSource = new LearnableWeightsNode<Vector>(biases),
            },
        };
    }

    [Benchmark(Baseline = true)]
    public Vector Module()
    {
        var output = module.Forward(input, snapshot);
        module.Backward(output, snapshot, gradients);
        return output;
    }

    [Benchmark]
    public Vector Node()
    {
        inputNode.Set(input);
        var output = node.Evaluate();
        node.Backward(output);
        return output;
    }
}