using ML.Core.Attributes;

namespace ML.Core.Modules;

[GeneratedModule]
public sealed partial class PerceptronModule : IHiddenModule<Vector>, IHiddenModule<Vector, PerceptronModule.Snapshot, PerceptronModule.Gradients>
{
    [Weights] public required Matrix Weights { get; init; }
    [Weights] public required Vector Biases { get; init; }
    [SubModule] public required IHiddenModule<Vector> Activation { get; init; }

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        snapshot.Input = input;
        Weights.MultiplyTo(snapshot.Input, snapshot.Weighted);
        snapshot.Weighted.AddTo(Biases, snapshot.Biased);
        snapshot.Activated = Activation.Forward(snapshot.Biased, snapshot.Activation);
        return snapshot.Activated;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        var biasedGradient = Activation.Backward(outputGradient, snapshot.Activation, gradients.Activation);
    }

    partial class Snapshot
    {
        public Vector Input { get; set; }
        public Vector Weighted { get; } = Vector.OfSize(module.Biases);
        public Vector Biased { get; } = Vector.OfSize(module.Biases);
        public Vector Activated { get; set; }
    }
}
