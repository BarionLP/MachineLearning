namespace ML.Core.Modules;

public interface IModule
{
    public ulong ParameterCount { get; }

    public IModuleSnapshot CreateSnapshot();
    public IModuleGradients CreateGradients();
}

public interface IModule<TArch> : IModule
{
    public TArch Backward(TArch outputGradient, IModuleSnapshot snapshot, IModuleGradients gradients);
}

public interface IModule<TArch, TSnapshot, TGradients> : IModule<TArch>
    where TSnapshot : IModuleSnapshot
    where TGradients : IModuleGradients
{
    public TArch Backward(TArch outputGradient, TSnapshot snapshot, TGradients gradients);
    TArch IModule<TArch>.Backward(TArch outputGradient, IModuleSnapshot snapshot, IModuleGradients gradients)
        => Backward(outputGradient, Guard.Is<TSnapshot>(snapshot), Guard.Is<TGradients>(gradients));

    public new TSnapshot CreateSnapshot();
    public new TGradients CreateGradients();

    IModuleSnapshot IModule.CreateSnapshot() => CreateSnapshot();
    IModuleGradients IModule.CreateGradients() => CreateGradients();
}

public interface IModuleSnapshot;

public interface IModuleGradients
{
    public void Add(IModuleGradients other);
    public void Reset();
}

public interface IModuleGradients<TSelf> : IModuleGradients where TSelf : IModuleGradients
{
    public void Add(TSelf other);
    void IModuleGradients.Add(IModuleGradients other) => Add(Guard.Is<TSelf>(other));
}

public sealed class EmptyModuleGradients() : IModuleGradients<EmptyModuleGradients>
{
    public static EmptyModuleGradients Instance => field ??= new();
    public EmptyModuleGradients(object? _) : this() { }

    public void Add(EmptyModuleGradients other) { }
    public void Reset() { }
}