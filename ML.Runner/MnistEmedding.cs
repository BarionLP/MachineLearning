using ML.Core.Attributes;
using ML.Core.Modules;
using ML.Core.Training;

namespace ML.Runner;

[GeneratedModule]
public sealed partial class MnistInput : IInputModule<double[], Vector, EmptyModuleData, EmptyModuleData>
{
    public static MnistInput Instance => field ??= new MnistInput();
    public Vector Forward(double[] input, EmptyModuleData snapshot) => Vector.Of([.. input.Select(v => (Weight)v)]);

    public Vector Backward(Vector outputGradient, EmptyModuleData snapshot, EmptyModuleData gradients) => outputGradient;
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