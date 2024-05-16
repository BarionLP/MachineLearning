using MachineLearning.Domain.Activation;

namespace MachineLearning.Serialization.Activation;

public static class ActivationMethodSerializer<TData> where TData : struct, IEquatable<TData>, IFormattable
{
    private static readonly Dictionary<string, IActivationMethod<TData>> _registry = [];

    public static void Register(string key, IActivationMethod<TData> activationMethod) => _registry.Add(key, activationMethod);

    public static void Write(BinaryWriter writer, IActivationMethod<TData> data) => writer.Write(_registry.GetKey(data));
    public static IActivationMethod<TData> Read(BinaryReader reader) => _registry[reader.ReadString()];
}

public static class ActivationMethodSerializer
{
    public static void RegisterDefaults()
    {
        ActivationMethodSerializer<double>.Register("sigmoid", SigmoidActivation.Instance);
        ActivationMethodSerializer<double>.Register("softmax", SoftmaxActivation.Instance);
        ActivationMethodSerializer<double>.Register("relu", ReLUActivation.Instance);
        ActivationMethodSerializer<double>.Register("leakyrelu", LeakyReLUActivation.Instance);
        ActivationMethodSerializer<double>.Register("tanh", TanhActivation.Instance);
    }
}
