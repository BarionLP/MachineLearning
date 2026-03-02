using ML.Core.Attributes;

namespace ML.Core.Modules;

[GeneratedModule]
public sealed partial class PerceptronModule(int inputNodes, int outputNodes) : IHiddenModule<Vector>
{
    [Property] public int InputNodes => Weights.ColumnCount;
    [Property] public int OutputNodes => Weights.RowCount;
    [Weights] public Matrix Weights { get; } = Matrix.Create(outputNodes, inputNodes);
    [Weights] public Vector Biases { get; } = Vector.Create(outputNodes);
    [SubModule] public required IHiddenModule<Vector> Activation { get; init; }

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count == inputNodes);
        snapshot.Input = input;
        Weights.MultiplyTo(snapshot.Input, snapshot.Weighted);
        snapshot.Weighted.AddTo(Biases, snapshot.Biased);
        snapshot.Activated = Activation.Forward(snapshot.Biased, snapshot.Activation);
        return snapshot.Activated;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        var biasedGradient = Activation.Backward(outputGradient, snapshot.Activation, gradients.Activation);
        biasedGradient.PointwiseMultiplyToSelf(outputGradient); // TODO: this might be bad, we no longer own biasedGradient
        gradients.Biases.AddToSelf(biasedGradient);
        VectorHelper.MultiplyToMatrixAddTo(outputGradient, snapshot.Input, gradients.Weights);
        Weights.MultiplyTransposedTo(outputGradient, snapshot.InputGradient);
        return snapshot.InputGradient;
    }

    partial class Snapshot
    {
        public Vector Input { get; set; }
        public Vector Weighted { get; } = Vector.OfSize(module.Biases);
        public Vector Biased { get; } = Vector.OfSize(module.Biases);
        public Vector Activated { get; set; }
        public Vector InputGradient { get; } = Vector.Create(module.InputNodes);
    }
}
