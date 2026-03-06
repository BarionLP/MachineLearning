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