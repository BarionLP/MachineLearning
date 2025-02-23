using Ametrin.Guards;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public abstract class Optimizer
{
    public required Weight LearningRate { get; set; }
    public required ICostFunction CostFunction { get; init; }

    public virtual void Init() { }
    public virtual void OnBatchCompleted() { }
    public virtual void OnEpochCompleted() { }

    protected abstract LayerOptimizerRegistry RegistryGetter { get; }
    public ILayerOptimizer CreateLayerOptimizer(ILayer layer)
    {
        if (RegistryGetter.TryGetValue(layer.GetType(), out var factory))
        {
            return factory(this, layer);
        }

        throw new NotImplementedException($"No known {GetType().Name} for {layer.GetType().Name}");
    }
}

public class LayerOptimizerRegistry : Dictionary<Type, Func<Optimizer, ILayer, ILayerOptimizer>>;
public sealed class LayerOptimizerRegistry<TOptimizer> : LayerOptimizerRegistry
{
    public void Register<TLayer>(Func<TOptimizer, TLayer, ILayerOptimizer> factory) where TLayer : ILayer
        => Add(typeof(TLayer), (op, layer) => factory(Guard.Is<TOptimizer>(op), Guard.Is<TLayer>(layer)));

    public void RegisterEmpty<TLayer>() where TLayer : ILayer
        => Add(typeof(TLayer), static (_, _) => EmptyLayerOptimizer.Instance);
}