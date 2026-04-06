using ML.Core.Modules.Activations;

namespace ML.Core.Modules.Initialization;

public interface IModuleInitializer
{
    public IModule Init(IModule module);
}

public interface IModuleInitializer<TModule> : IModuleInitializer
    where TModule : IModule
{
    public TModule Init(TModule module);
    IModule IModuleInitializer.Init(IModule module)
        => Init(Guard.Is<TModule>(module));
}

public sealed class EmptyModuleInitializer : IModuleInitializer
{
    public static EmptyModuleInitializer Instance => field ??= new();
    public IModule Init(IModule module) => module;
}

public static class InitializationHelper
{
    public static Weight GetKaimingGain(IActivationModule n) => n switch
    {
        // SigmoidActivation => 1,
        // TanhActivation => 5 / 3,
        // ReLUActivation => Weight.Sqrt(2f),
        LeakyReLUActivation l => Weight.Sqrt(2 / (1 + l.Alpha * l.Alpha)),
        GeLUActivation => Weight.Sqrt(2),   // common approximation
        // Nonlinearity.Swish => Weight.Sqrt(2.0),  // reasonable default
        _ => throw new NotImplementedException(),
    };
}