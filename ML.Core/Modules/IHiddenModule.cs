namespace ML.Core.Modules;

public interface IHiddenModule<TArch> : IInputModule<TArch, TArch>;

public interface IHiddenModule<TArch, TSnapshot, TGradients> : IHiddenModule<TArch>, IModule<TArch, TSnapshot, TGradients>
    where TSnapshot : IModuleSnapshot
    where TGradients : IModuleGradients
{
    public TArch Forward(TArch input, TSnapshot snapshot);
    TArch IInputModule<TArch, TArch>.Forward(TArch input, IModuleSnapshot snapshot)
        => Forward(input, Guard.Is<TSnapshot>(snapshot));
}