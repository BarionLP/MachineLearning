using ML.Core.Attributes;
using ML.Core.Modules;

namespace ML.Runner.Samples.Mnist;

[GeneratedModule]
public sealed partial class MnistInput(int outputNodes) : IInputModule<double[], Vector, MnistInput.Snapshot, EmptyModuleData>
{
    public static MnistInput Instance => field ??= new MnistInput(MnistImage.SIZE * MnistImage.SIZE);

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
