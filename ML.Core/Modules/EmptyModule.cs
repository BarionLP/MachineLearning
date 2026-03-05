using ML.Core.Attributes;

namespace ML.Core.Modules;

[GeneratedModule]
public sealed partial class EmptyModule() : IHiddenModule<Vector, EmptyModuleData, EmptyModuleData>
{
    public Vector Forward(Vector input, EmptyModuleData snapshot) => input;
    public Vector Backward(Vector outputGradient, EmptyModuleData snapshot, EmptyModuleData gradients) => outputGradient;
}