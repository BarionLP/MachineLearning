using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Adam;

public sealed class EmptyAdamOptimizer(ILayer layer) : ILayerOptimizer
{
    public ILayer Layer { get; } = layer;
    public ICostFunction CostFunction => null!;

    public void Apply(int dataCounter) { }
    public void FullReset() { }
    public void GradientCostReset() { }
    public void Update(Vector nodeValues, ILayerSnapshot snapshot) { }
}