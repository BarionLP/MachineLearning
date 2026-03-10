using ML.Core.Attributes;
using ML.Core.Modules;

namespace ML.Runner.Mnist;

[GeneratedModule]
public sealed partial class MnistInput(int outputNodes) : IInputModule<double[], Vector, MnistInput.Snapshot, EmptyModuleData>
{
    public static MnistInput Instance => field ??= new MnistInput(784);

    [Property] public int OutputNodes { get; } = outputNodes;

    public Vector Forward(double[] input, Snapshot snapshot)
    {
        foreach (var i in input.IndexRange)
        {
            snapshot.Output[i] = (Weight)input[i];
        }
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, EmptyModuleData gradients) => outputGradient;

    public sealed class Snapshot(MnistInput module) : IModuleSnapshot
    {
        public Vector Output { get; } = Vector.Create(module.OutputNodes);
        public void Dispose() { }
    }
}

[GeneratedModule]
public sealed partial class MnistOuput(ImmutableArray<int> _nodeMapping) : IOutputModule<Vector, int, EmptyModuleData, EmptyModuleData>
{
    public static MnistOuput Instance => field ??= new MnistOuput([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    private readonly ImmutableArray<int> _nodeMapping = _nodeMapping;

    public (int Output, float Confidence, Vector Weights) Forward(Vector input, EmptyModuleData snapshot)
    {
        var index = input.MaximumIndex();
        return (_nodeMapping[index], input[index], input);
    }

    public Vector Backward(Vector outputGradient, EmptyModuleData snapshot, EmptyModuleData gradients) => outputGradient;
}