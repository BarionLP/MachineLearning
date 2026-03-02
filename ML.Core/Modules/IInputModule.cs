namespace ML.Core.Modules;

public interface IInputModule<in TIn, TArch> : IModule<TArch>
{
    public TArch Forward(TIn input, IModuleSnapshot snapshot);
}

public interface IInputModule<TIn, TArch, TSnapshot, TGradients> : IInputModule<TIn, TArch>, IModule<TArch, TSnapshot, TGradients>
    where TSnapshot : IModuleSnapshot
    where TGradients : IModuleGradients
{
    public TArch Forward(TIn input, TSnapshot snapshot);
    TArch IInputModule<TIn, TArch>.Forward(TIn input, IModuleSnapshot snapshot)
        => Forward(input, Guard.Is<TSnapshot>(snapshot));
}
