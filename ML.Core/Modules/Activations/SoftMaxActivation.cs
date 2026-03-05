using System.Runtime.InteropServices;
using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

[GeneratedModule]
public sealed partial class SoftMaxActivation(int inputNodes) : IHiddenModule<Vector, SoftMaxActivation.Snapshot, EmptyModuleData>, IActivationModule
{
    [Property] public int InputNodes { get; } = inputNodes;

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count == InputNodes);
        snapshot.Input = input;
        snapshot.Input.SoftMaxTo(snapshot.Output);
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, EmptyModuleData gradients)
    {
        var dot = snapshot.Output.Dot(outputGradient);
        outputGradient.SubtractPointwiseTo(dot, snapshot.InputGradient);
        snapshot.InputGradient.PointwiseMultiplyToSelf(snapshot.Output);
        NumericsDebug.AssertValidNumbers(snapshot.InputGradient);
        return snapshot.InputGradient;
    }

    public sealed class Snapshot(SoftMaxActivation module) : IModuleSnapshot
    {
        public Vector Input { get; set; }
        public Vector Output { get; } = Vector.Create(module.InputNodes);
        public Vector InputGradient { get; } = Vector.Create(module.InputNodes);
    }
}