using Ametrin.Serializer;
using ML.Core.Attributes;
using ML.Core.Converters;
using ML.Core.Modules.Activations;
using ML.Core.Modules.Initialization;

namespace ML.Core.Modules;

[GeneratedModule, GenerateSerializer]
public sealed partial class PerceptronModule(int inputNodes, int outputNodes) : IHiddenModule<Vector>
{
    [Property] public int InputNodes => Weights.ColumnCount;
    [Property] public int OutputNodes => Weights.RowCount;
    [Weights, Serialize(Converter: typeof(VectorConverter))] public Matrix Weights { get; } = Matrix.Create(outputNodes, inputNodes);
    [Weights, Serialize(Converter: typeof(VectorConverter))] public Vector Biases { get; } = Vector.Create(outputNodes);
    [SubModule] public required IActivationModule<Vector> Activation { get; init; }

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count == inputNodes);
        snapshot.Input = input;
        Weights.MultiplyTo(snapshot.Input, snapshot.Weighted);
        snapshot.Weighted.AddTo(Biases, snapshot.Biased);
        return Activation.Forward(snapshot.Biased, snapshot.Activation);
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        var biasedGradient = Activation.Backward(outputGradient, snapshot.Activation, gradients.Activation);
        gradients.Biases.AddToSelf(biasedGradient);
        VectorHelper.MultiplyToMatrixAddTo(biasedGradient, snapshot.Input, gradients.Weights);
        Weights.MultiplyTransposedTo(biasedGradient, snapshot.InputGradient);
        return snapshot.InputGradient;
    }

    partial class Snapshot
    {
        public Vector Input { get; set; }
        public Vector Weighted { get; } = Vector.OfSize(module.Biases);
        public Vector Biased { get; } = Vector.OfSize(module.Biases);
        public Vector InputGradient { get; } = Vector.Create(module.InputNodes);
    }

    [GeneratedAdam(typeof(PerceptronModule))]
    public sealed partial class Adam;

    /// <summary>
    /// suited for (Leaky)ReLU<br/>
    /// not suited for SoftMax/Sigmoid
    /// </summary>
    public sealed class KaimingInitializer(IActivationModule activation) : IModuleInitializer<PerceptronModule>
    {
        public Random Random { get; init; } = Random.Shared;
        private readonly Weight gain = InitializationHelper.GetKaimingGain(activation);
        public PerceptronModule Init(PerceptronModule module)
        {
            Debug.Assert(module.Activation is not SoftMaxActivation);
            module.Weights.KaimingNormal(gain, Random);
            module.Biases.Normal(0, 0.1f, Random);
            return module;
        }
    }

    /// <summary>
    /// suited for SoftMax/Sigmoid<br/>
    /// not suited for (Leaky)ReLU
    /// </summary>
    public sealed class XavierInitializer : IModuleInitializer<PerceptronModule>
    {
        public static XavierInitializer Instance => field ??= new();
        public Random Random { get; init; } = Random.Shared;
        public PerceptronModule Init(PerceptronModule module)
        {
            Debug.Assert(module.Activation is not LeakyReLUActivation);
            module.Weights.XavierUniform(Random);
            module.Biases.Normal(0, 0.1f, Random);
            return module;
        }
    }
}
