using ML.Core.Modules;

namespace ML.Core.Training;

public interface IModuleOptimizer
{
    public void Apply(IModuleGradients gradients);
    public void FullReset();
};

public interface IModuleOptimizer<TGradients> : IModuleOptimizer
    where TGradients : IModuleGradients<TGradients>
{
    public void Apply(TGradients gradients);
    void IModuleOptimizer.Apply(IModuleGradients gradients)
        => Apply(Guard.Is<TGradients>(gradients));
}

public sealed class EmptyModuleOptimizer : IModuleOptimizer<EmptyModuleGradients>
{
    public static EmptyModuleOptimizer Instance { get; } = new();

    public void Apply(EmptyModuleGradients gradients) { }
    public void FullReset() { }
}