namespace ML.Core.Modules;

public interface IOutputModule<TArch, TOut> : IModule<TArch>
{
    public (TOut Output, Weight Confidence, TArch Weights) Forward(TArch input, IModuleSnapshot snapshot);
}

public interface IOutputModule<TArch, TOut, TSnapshot, TGradients> : IOutputModule<TArch, TOut>, IModule<TArch, TSnapshot, TGradients>
    where TSnapshot : IModuleSnapshot
    where TGradients : IModuleGradients
{
    public (TOut Output, Weight Confidence, TArch Weights) Forward(TArch input, TSnapshot snapshot);
    (TOut Output, Weight Confidence, TArch Weights) IOutputModule<TArch, TOut>.Forward(TArch input, IModuleSnapshot snapshot)
        => Forward(input, Guard.Is<TSnapshot>(snapshot));
}
