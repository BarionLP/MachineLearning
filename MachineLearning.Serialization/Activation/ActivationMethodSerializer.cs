using MachineLearning.Model.Activation;

namespace MachineLearning.Serialization.Activation;

public static class ActivationMethodSerializer
{
    private static readonly Dictionary<string, IActivationFunction> _legacyRegistry = [];
    private static readonly Dictionary<Type, string> _registry = [];
    private static readonly Dictionary<string, Func<BinaryReader, IActivationFunction>> _factory = [];
    private static readonly Dictionary<Type, (string key, uint version)> _registryV3 = [];
    private static readonly Dictionary<(string key, uint version), Func<BinaryReader, IActivationFunction>> _factoryV3 = [];

    public static void RegisterV2<T>(string key, Func<BinaryReader, IActivationFunction> factory) where T : IActivationFunction
    {
        _registry.Add(typeof(T), key);
        _factory.Add(key, factory);
    }
    
    public static void Register<T>(string key, uint version, Func<BinaryReader, IActivationFunction> factory) where T : IActivationFunction
    {
        _registryV3.Add(typeof(T), (key, version));
        _factoryV3.Add((key, version), factory);
    }

    public static void RegisterV1(string key, IActivationFunction instance) => _legacyRegistry.Add(key, instance);

    public static void WriteV1(BinaryWriter writer, IActivationFunction data) => writer.Write(_registry[data.GetType()]);
    public static IActivationFunction ReadV1(BinaryReader reader) => _legacyRegistry[reader.ReadString()];
    public static void WriteV3(BinaryWriter writer, IActivationFunction data)
    {
        var (key, version) = _registryV3[data.GetType()];
        writer.Write(key);
        writer.Write(version);

        // Update reader when changing
        switch (data)
        {
            case LeakyReLUActivation leakyReLU:
                writer.Write(leakyReLU.Alpha);
                break;


            case SoftMaxActivation:
            case TanhActivation:
            case SigmoidActivation:
            case ReLUActivation:
                break;
                
            default:
                throw new NotImplementedException();
        }
    }

    public static IActivationFunction ReadV2(BinaryReader reader) => _factory[reader.ReadString()](reader);
    public static IActivationFunction ReadV3(BinaryReader reader) => _factoryV3[(reader.ReadString(), reader.ReadUInt32())](reader);

    public static void RegisterDefaults()
    {
        Register<SigmoidActivation>("sigmoid", 1, reader => SigmoidActivation.Instance);
        Register<SoftMaxActivation>("softmax", 1, reader => SoftMaxActivation.Instance);
        Register<ReLUActivation>("relu", 1, reader => ReLUActivation.Instance);
        Register<LeakyReLUActivation>("leakyrelu", 1, reader => new LeakyReLUActivation(reader.ReadSingle()));
        Register<TanhActivation>("tanh", 1, reader => TanhActivation.Instance);
        
        RegisterV2<SigmoidActivation>("sigmoid", reader => SigmoidActivation.Instance);
        RegisterV2<SoftMaxActivation>("softmax", reader =>
        {
            reader.ReadDouble(); //ignore temperature
            return new SoftMaxActivation();
        });
        RegisterV2<ReLUActivation>("relu", reader => ReLUActivation.Instance);
        RegisterV2<LeakyReLUActivation>("leakyrelu", reader => new LeakyReLUActivation((float) reader.ReadDouble()));
        RegisterV2<TanhActivation>("tanh", reader => TanhActivation.Instance);

        RegisterV1("sigmoid", SigmoidActivation.Instance);
        RegisterV1("softmax", SoftMaxActivation.Instance);
        RegisterV1("relu", ReLUActivation.Instance);
        RegisterV1("leakyrelu", LeakyReLUActivation.Instance);
        RegisterV1("tanh", TanhActivation.Instance);
    }
}
