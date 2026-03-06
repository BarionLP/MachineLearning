namespace ML.Core.Modules.Activations;

public interface IActivationModule : IModule;
public interface IActivationModule<TArch> : IActivationModule, IHiddenModule<TArch>;
public interface IActivationModule<TArch, TSnapshot> : IActivationModule<TArch>, IHiddenModule<TArch, TSnapshot, EmptyModuleData> where TSnapshot : IModuleSnapshot;
