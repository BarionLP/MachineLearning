using Simple.Network.Activation;

namespace Simple.Serialization.Activation;

public static class ActivationMethodSerializer<TData>{
    private static readonly Dictionary<string, IActivationMethod<TData>> _registry = [];

    public static void Register(string key, IActivationMethod<TData> activationMethod) => _registry.Add(key, activationMethod);

    public static void Write(BinaryWriter writer, IActivationMethod<TData> data) => writer.Write(_registry.GetKey(data));
    public static IActivationMethod<TData> Read(BinaryReader reader) => _registry[reader.ReadString()];
}

public static class ActivationMethodSerializer{
    public static void RegisterDefaults(){
        ActivationMethodSerializer<Number>.Register("sigmoid", SigmoidActivation.Instance);
        ActivationMethodSerializer<Number>.Register("softmax", SoftmaxActivation.Instance);
        ActivationMethodSerializer<Number>.Register("relu", ReLUActivation.Instance);
        ActivationMethodSerializer<Number>.Register("leakyrelu", LeakyReLUActivation.Instance);
        ActivationMethodSerializer<Number>.Register("tanh", TanhActivation.Instance);
    }
}
