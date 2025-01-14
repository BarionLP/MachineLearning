using Ametrin.Guards;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public abstract class Optimizer
{
    private readonly Dictionary<Type, Func<ILayer, ILayerOptimizer>> _registry = [];

    public required Weight LearningRate { get; set; }
    public required ICostFunction CostFunction { get; init; }

    public virtual void Init() { }
    public virtual void OnBatchCompleted() { }
    public virtual void OnEpochCompleted() { }

    protected Optimizer()
    {
        RegisterEmpty<TokenOutputLayer>();
        RegisterEmpty<EncodedEmbeddingLayer>();
    }

    public ILayerOptimizer CreateLayerOptimizer(ILayer layer)
    {
        if(_registry.TryGetValue(layer.GetType(), out var factory))
        {
            return factory(layer);
        }

        throw new NotImplementedException($"No known {GetType().Name} for {layer.GetType().Name}");
    }


    public void Register<TLayer>(Func<TLayer, ILayerOptimizer> factory) where TLayer : ILayer
        => _registry.Add(typeof(TLayer), (layer) => factory(Guard.Is<TLayer>(layer)));
    
    public void RegisterEmpty<TLayer>() where TLayer : ILayer 
        => _registry.Add(typeof(TLayer), static (_) => EmptyLayerOptimizer.Instance);
}