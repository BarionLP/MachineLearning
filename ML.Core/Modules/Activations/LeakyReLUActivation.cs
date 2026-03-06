using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

[GeneratedModule]
public sealed partial class LeakyReLUActivation(int inputNodes, Weight alpha = 0.01f) : IActivationModule<Vector, LeakyReLUActivation.Snapshot>
{
    [Property] public int InputNodes { get; } = inputNodes;
    [Property] public Weight Alpha { get; } = alpha;

    public Weight Activate(Weight input) => input > 0 ? input : Alpha * input;
    public SimdVector Activate(SimdVector input) => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), input, input * Alpha);

    public Weight Derivative(Weight input) => input > 0 ? 1 : Alpha;
    public SimdVector Derivative(SimdVector input) => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), SimdVector.One, new SimdVector(Alpha));


    public Vector Forward(Vector input, Snapshot snapshot)
    {
        snapshot.Input = input;
        snapshot.Input.MapTo(Activate, Activate, snapshot.Output);
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, EmptyModuleData gradients)
    {
        snapshot.Input.MapTo(Derivative, Derivative, snapshot.InputGradient);
        snapshot.InputGradient.PointwiseMultiplyToSelf(outputGradient);
        return snapshot.InputGradient;
    }

    public sealed class Snapshot(LeakyReLUActivation module) : IModuleSnapshot
    {
        public Vector Input { get; set; }
        public Vector Output { get; } = Vector.Create(module.InputNodes);
        public Vector InputGradient { get; } = Vector.Create(module.InputNodes);
    }
}