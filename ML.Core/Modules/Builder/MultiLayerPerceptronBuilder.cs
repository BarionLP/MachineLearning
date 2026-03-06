using ML.Core.Modules.Activations;

namespace ML.Core.Modules.Builder;

public sealed class MultiLayerPerceptronBuilder
{
    private readonly List<(int input, int output, IActivationModule<Vector> activation)> layers = [];
    private int nextInput;
    public static MultiLayerPerceptronBuilder Create(int inputNodes) => new() { nextInput = inputNodes };

    public MultiLayerPerceptronBuilder AddLayer(int outputNodes, IActivationModule<Vector> activation)
    {
        layers.Add((nextInput, outputNodes, activation));
        nextInput = outputNodes;
        return this;
    }

    public MultiLayerPerceptronBuilder AddLayer(int outputNodes, Func<int, int, IActivationModule<Vector>> activation)
    {
        layers.Add((nextInput, outputNodes, activation.Invoke(nextInput, outputNodes)));
        nextInput = outputNodes;
        return this;
    }

    public SequenceModule<Vector> Build() => new()
    {
        Inner = [.. layers.Select(d => new PerceptronModule(d.input, d.output) { Activation = d.activation })],
    };
}