using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class SoftMaxActivation : IActivationModule<Vector, SimpleModuleSnapshot>
{
    public static SoftMaxActivation Instance => field ??= new();
    public Vector Forward(Vector input, SimpleModuleSnapshot snapshot)
    {
        snapshot.Input = input;
        snapshot.Input.SoftMaxTo(snapshot.Output);
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, SimpleModuleSnapshot snapshot, EmptyModuleData gradients)
    {
        var dot = snapshot.Output.Dot(outputGradient);
        outputGradient.SubtractPointwiseTo(dot, snapshot.InputGradient);
        snapshot.InputGradient.PointwiseMultiplyToSelf(snapshot.Output);
        NumericsDebug.AssertValidNumbers(snapshot.InputGradient);
        return snapshot.InputGradient;
    }
}
