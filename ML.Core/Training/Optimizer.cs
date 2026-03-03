using ML.Core.Modules;

namespace ML.Core.Training;

public abstract class Optimizer
{
    public required Weight LearningRate { get; set; }

    public virtual void Init() { }
    public virtual void OnBatchCompleted() { }
    public virtual void OnEpochCompleted() { }

    protected abstract ModuleOptimizerRegistry RegistryGetter { get; }
    public IModuleOptimizer CreateModuleOptimizer(IModule module)
    {
        if (RegistryGetter.TryGetValue(module.GetType(), out var factory))
        {
            return factory(this, module);
        }

        throw new NotImplementedException($"No known {GetType().Name} for {module.GetType().Name}");
    }
}

public class ModuleOptimizerRegistry : Dictionary<Type, Func<Optimizer, IModule, IModuleOptimizer>>;
public sealed class ModuleOptimizerRegistry<TOptimizer> : ModuleOptimizerRegistry
{
    public void Register<TModule>(Func<TOptimizer, TModule, IModuleOptimizer> factory) where TModule : IModule
        => Add(typeof(TModule), (op, layer) => factory(Guard.Is<TOptimizer>(op), Guard.Is<TModule>(layer)));

    public void RegisterEmpty<TModule>() where TModule : IModule
        => Add(typeof(TModule), static (_, _) => EmptyModuleOptimizer.Instance);
}