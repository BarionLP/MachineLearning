using System.Diagnostics.CodeAnalysis;
using ML.Core.Attributes;
using ML.Core.Modules.Activations;
using ML.Core.Modules.Initialization;

namespace ML.Core.Modules;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class LinearVectorModule : IHiddenModule<Vector>
{
    [Weights] public Matrix Weights { get; }
    [Weights] public Vector Biases { get; }

    public int InputNodes => Weights.ColumnCount;
    public int OutputNodes => Weights.RowCount;

    public LinearVectorModule(int inputNodes, int outputNodes)
    {
        Weights = Matrix.Create(outputNodes, inputNodes);
        Biases = Vector.Create(outputNodes);
    }

    [SetsRequiredMembers]
    public LinearVectorModule(Matrix weights, Vector biases)
    {
        Debug.Assert(weights.RowCount == biases.Count);
        Weights = weights;
        Biases = biases;
    }

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count == InputNodes);
        snapshot.Input = input;
        Weights.MultiplyTo(snapshot.Input, snapshot.Weighted);
        snapshot.Weighted.AddTo(Biases, snapshot.Biased);
        return snapshot.Biased;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        gradients.Biases.AddToSelf(outputGradient);
        VectorHelper.MultiplyToMatrixAddTo(outputGradient, snapshot.Input, gradients.Weights);
        Weights.MultiplyTransposedTo(outputGradient, snapshot.InputGradient);
        NumericsDebug.AssertValidNumbers(snapshot.InputGradient);
        return snapshot.InputGradient;
    }

    partial class Snapshot
    {
        public Vector Input { get; set; }
        public Vector Weighted { get; } = Vector.OfSize(module.Biases);
        public Vector Biased { get; } = Vector.OfSize(module.Biases);
        public Vector InputGradient { get; } = Vector.Create(module.InputNodes);
    }

    [GeneratedAdam(typeof(LinearVectorModule))]
    public sealed partial class Adam;
}

public static class LinearModule
{
    /// <summary>
    /// suited for (Leaky)ReLU<br/>
    /// not suited for SoftMax/Sigmoid
    /// </summary>
    public sealed class KaimingInitializer(IActivationModule activation) : IModuleInitializer<LinearVectorModule>, IModuleInitializer<LinearMatrixModule>
    {
        public Random Random { get; init; } = Random.Shared;
        private readonly Weight gain = InitializationHelper.GetKaimingGain(activation);
        public LinearVectorModule Init(LinearVectorModule module)
        {
            Init(module.Weights, module.Biases);
            return module;
        }

        public LinearMatrixModule Init(LinearMatrixModule module)
        {
            Init(module.Weights, module.Biases);
            return module;
        }

        public void Init(Matrix weights, Vector biases)
        {
            // Debug.Assert(module.Activation is not SoftMaxActivation);
            weights.KaimingNormal(gain, Random);
            biases.Normal(0, 0.1f, Random);
        }

        IModule IModuleInitializer.Init(IModule module)
            => module is LinearVectorModule vector ? Init(vector) : module is LinearMatrixModule matrix ? Init(matrix) : throw new UnreachableException();
    }

    /// <summary>
    /// suited for SoftMax/Sigmoid<br/>
    /// not suited for (Leaky)ReLU
    /// </summary>
    public sealed class XavierInitializer : IModuleInitializer<LinearVectorModule>, IModuleInitializer<LinearMatrixModule>
    {
        public static XavierInitializer Instance => field ??= new();
        public Random Random { get; init; } = Random.Shared;
        public LinearVectorModule Init(LinearVectorModule module)
        {
            Init(module.Weights, module.Biases);
            return module;
        }

        public LinearMatrixModule Init(LinearMatrixModule module)
        {
            Init(module.Weights, module.Biases);
            return module;
        }

        public void Init(Matrix weights, Vector biases)
        {
            weights.XavierUniform(Random);
            biases.Normal(0, 0.1f, Random);
        }

        IModule IModuleInitializer.Init(IModule module)
            => module is LinearVectorModule vector ? Init(vector) : module is LinearMatrixModule matrix ? Init(matrix) : throw new UnreachableException();
    }
}