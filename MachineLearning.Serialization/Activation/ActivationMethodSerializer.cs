using MachineLearning.Domain.Activation;

namespace MachineLearning.Serialization.Activation;

public static class ActivationMethodSerializer
{
    private static readonly Dictionary<string, IActivationMethod> _registry = [];

    public static void Register(string key, IActivationMethod activationMethod) => _registry.Add(key, activationMethod);

    public static void Write(BinaryWriter writer, IActivationMethod data) => writer.Write(_registry.GetKey(data));
    public static IActivationMethod Read(BinaryReader reader) => _registry[reader.ReadString()];

    public static void RegisterDefaults()
    {
        Register("sigmoid", SigmoidActivation.Instance);
        Register("softmax", SoftmaxActivation.Instance);
        Register("relu", ReLUActivation.Instance);
        Register("leakyrelu", LeakyReLUActivation.Instance);
        Register("tanh", TanhActivation.Instance);
    }
}
