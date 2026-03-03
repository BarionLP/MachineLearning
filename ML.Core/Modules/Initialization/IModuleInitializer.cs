namespace ML.Core.Modules.Initialization;

public interface IModuleInitializer
{
    public void Init(IModule module);
}

public interface IModuleInitializer<TModule> : IModuleInitializer
    where TModule : IModule
{
    public void Init(TModule module);
    void IModuleInitializer.Init(IModule module)
        => Init(Guard.Is<TModule>(module));
}

public sealed class EmptyModuleInitializer : IModuleInitializer
{
    public static EmptyModuleInitializer Instance => field ??= new();
    public void Init(IModule module) { }
}