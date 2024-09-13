using MachineLearning.Model.Activation;

namespace MachineLearning.Serialization.Activation;

public static class ActivationMethodSerializer
{
    private static readonly Dictionary<string, IActivationFunction> _legacyRegistry = [];
    private static readonly Dictionary<Type, string> _registry = [];
    private static readonly Dictionary<string, Func<BinaryReader, IActivationFunction>> _factory = [];

    public static void Register<T>(string key, Func<BinaryReader, IActivationFunction> factory) where T : IActivationFunction
    {
        _registry.Add(typeof(T), key);
        _factory.Add(key, factory);
    }

    public static void RegisterLegacy(string key, IActivationFunction instance) => _legacyRegistry.Add(key, instance);

    public static void WriteV1(BinaryWriter writer, IActivationFunction data) => writer.Write(_registry[data.GetType()]);
    public static IActivationFunction ReadV1(BinaryReader reader) => _legacyRegistry[reader.ReadString()];
    public static void WriteV2(BinaryWriter writer, IActivationFunction data)
    {
        writer.Write(_registry[data.GetType()]);
        // Update reader when changing
        switch (data)
        {
            case LeakyReLUActivation leakyReLU:
                writer.Write(leakyReLU.Alpha);
                break;

            case SoftMaxActivation softmax:
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

    public static IActivationFunction ReadV2(BinaryReader reader) => _factory[reader.ReadString()](reader);

    public static void RegisterDefaults()
    {
        RegisterLegacy("sigmoid", SigmoidActivation.Instance);
        RegisterLegacy("softmax", SoftMaxActivation.Instance);
        RegisterLegacy("relu", ReLUActivation.Instance);
        RegisterLegacy("leakyrelu", LeakyReLUActivation.Instance);
        RegisterLegacy("tanh", TanhActivation.Instance);
        
        Register<SigmoidActivation>("sigmoid", reader => SigmoidActivation.Instance);
        Register<SoftMaxActivation>("softmax", reader => new SoftMaxActivation(reader.ReadDouble()));
        Register<ReLUActivation>("relu", reader => ReLUActivation.Instance);
        Register<LeakyReLUActivation>("leakyrelu", reader => new LeakyReLUActivation(reader.ReadDouble()));
        Register<TanhActivation>("tanh", reader => TanhActivation.Instance);
    }
}
