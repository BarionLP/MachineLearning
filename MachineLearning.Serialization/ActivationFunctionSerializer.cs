using MachineLearning.Model.Activation;

namespace MachineLearning.Serialization;

public static class ActivationFunctionSerializer
{
    private static readonly Dictionary<Type, string> _registryV2 = [];
    private static readonly Dictionary<string, Func<BinaryReader, IActivationFunction>> _factoryV2 = [];
    private static readonly Dictionary<Type, (string key, uint version)> _registry = [];
    private static readonly Dictionary<(string key, uint version), Func<BinaryReader, IActivationFunction>> _factory = [];

    public static void RegisterV2<T>(string key, Func<BinaryReader, IActivationFunction> factory) where T : IActivationFunction
    {
        _registryV2.Add(typeof(T), key);
        _factoryV2.Add(key, factory);
    }
    
    public static void Register<T>(string key, uint version, Func<BinaryReader, IActivationFunction> factory) where T : IActivationFunction
    {
        _registry.Add(typeof(T), (key, version));
        _factory.Add((key, version), factory);
    }

    public static void Write(BinaryWriter writer, IActivationFunction data)
    {
        var (key, version) = _registry[data.GetType()];
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

    public static IActivationFunction ReadV2(BinaryReader reader) => _factoryV2[reader.ReadString()](reader);
    public static IActivationFunction Read(BinaryReader reader) => _factory[(reader.ReadString(), reader.ReadUInt32())](reader);

    static ActivationFunctionSerializer()
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
    }
}
