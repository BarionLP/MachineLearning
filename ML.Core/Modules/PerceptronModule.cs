using ML.Core.Attributes;
using ML.Core.Modules.Activations;
using ML.Core.Modules.Initialization;

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
        biasedGradient.PointwiseMultiplyTo(outputGradient, snapshot.BiasedGradient);
        gradients.Biases.AddToSelf(snapshot.BiasedGradient);
        VectorHelper.MultiplyToMatrixAddTo(snapshot.BiasedGradient, snapshot.Input, gradients.Weights);
        Weights.MultiplyTransposedTo(snapshot.BiasedGradient, snapshot.InputGradient);
        return snapshot.InputGradient;
    }

    partial class Snapshot
    {
        public Vector Input { get; set; }
        public Vector Weighted { get; } = Vector.OfSize(module.Biases);
        public Vector Biased { get; } = Vector.OfSize(module.Biases);
        public Vector Activated { get; set; }
        public Vector InputGradient { get; } = Vector.Create(module.InputNodes);
        public Vector BiasedGradient { get; } = Vector.OfSize(module.Biases);
    }

    [GeneratedAdam(typeof(PerceptronModule))]
    public sealed partial class Adam;

    public sealed class Initializer : IModuleInitializer<PerceptronModule>
    {
        public static Initializer Instance => field ??= new();
        public Random Random { get; init; } = Random.Shared;
        public void Init(PerceptronModule module)
        {
            switch (module.Activation)
            {
                case LeakyReLUActivation:
                    module.Weights.KaimingNormal((IActivationModule)module.Activation, Random);
                    break;

                case SoftMaxActivation:
                    module.Weights.XavierUniform(Random);
                    break;

                default:
                    throw new NotImplementedException($"no PerceptronModule init method for {module.Activation}");
            }
            module.Biases.Normal(0, 0.1f, Random);
        }
    }
}
