using MachineLearning.Domain.Activation;

namespace MachineLearning.Serialization.Activation;

public static class ActivationMethodSerializer
{
    private static readonly Dictionary<string, IActivationMethod> _legacyRegistry = [];
    private static readonly Dictionary<Type, string> _registry = [];
    private static readonly Dictionary<string, Func<BinaryReader, IActivationMethod>> _factory = [];

    public static void Register<T>(string key, Func<BinaryReader, IActivationMethod> factory) where T : IActivationMethod
    {
        _registry.Add(typeof(T), key);
        _factory.Add(key, factory);
    }

    public static void RegisterLegacy(string key, IActivationMethod instance) => _legacyRegistry.Add(key, instance);

    public static void Write(BinaryWriter writer, IActivationMethod data) => writer.Write(_registry[data.GetType()]);
    public static IActivationMethod Read(BinaryReader reader) => _legacyRegistry[reader.ReadString()];
    public static void WriteV3(BinaryWriter writer, IActivationMethod data)
    {
        writer.Write(_registry[data.GetType()]);
        // Update reader when changing
        switch (data)
        {
            case LeakyReLUActivation leakyReLU:
                writer.Write(leakyReLU.Alpha);
                break;

            case SoftmaxActivation softmax:
                writer.Write(softmax.Temperature);
                break;

            case TanhActivation:
            case SigmoidActivation:
            case ReLUActivation:
                break;
                
            default:
                throw new NotImplementedException();
        }
    }

    public static IActivationMethod ReadV3(BinaryReader reader) => _factory[reader.ReadString()](reader);

    public static void RegisterDefaults()
    {
        RegisterLegacy("sigmoid", SigmoidActivation.Instance);
        RegisterLegacy("softmax", SoftmaxActivation.Instance);
        RegisterLegacy("relu", ReLUActivation.Instance);
        RegisterLegacy("leakyrelu", LeakyReLUActivation.Instance);
        RegisterLegacy("tanh", TanhActivation.Instance);
        
        Register<SigmoidActivation>("sigmoid", reader => SigmoidActivation.Instance);
        Register<SoftmaxActivation>("softmax", reader => new SoftmaxActivation(reader.ReadDouble()));
        Register<ReLUActivation>("relu", reader => ReLUActivation.Instance);
        Register<LeakyReLUActivation>("leakyrelu", reader => new LeakyReLUActivation(reader.ReadDouble()));
        Register<TanhActivation>("tanh", reader => TanhActivation.Instance);
    }
}
