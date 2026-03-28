using ML.Core.Attributes;
using ML.Core.Modules.Activations;

namespace ML.Core.Modules;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class EmptyModule : IActivationModule<Vector, EmptyModuleData>
{
    public static EmptyModule Instance => field ??= new();
    public Vector Forward(Vector input, EmptyModuleData snapshot) => input;
    public Vector Backward(Vector outputGradient, EmptyModuleData snapshot, EmptyModuleData gradients) => outputGradient;
}