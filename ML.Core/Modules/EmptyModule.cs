using Ametrin.Serializer;
using ML.Core.Attributes;
using ML.Core.Modules.Activations;

namespace ML.Core.Modules;

[GeneratedModule]
public sealed partial class EmptyModule : IActivationModule<Vector, EmptyModuleData>, ISerializationConverter<EmptyModule>
{
    public static EmptyModule Instance => field ??= new();
    public Vector Forward(Vector input, EmptyModuleData snapshot) => input;
    public Vector Backward(Vector outputGradient, EmptyModuleData snapshot, EmptyModuleData gradients) => outputGradient;

    public static Result<EmptyModule, DeserializationError> TryReadValue(IAmetrinReader reader) => Instance;

    public static void WriteValue(IAmetrinWriter writer, EmptyModule value) { }
}