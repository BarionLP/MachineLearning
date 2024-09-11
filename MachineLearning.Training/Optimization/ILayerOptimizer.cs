using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public interface ILayerOptimizer : ILayerOptimizer<SimpleLayer, LayerSnapshot>;


public interface ILayerOptimizer<TLayer, TSnapshot> where TLayer : ILayer
{
    public TLayer Layer { get; }
    public ICostFunction CostFunction { get; }
    public void Update(Vector nodeValues, TSnapshot snapshot);
    public void Apply(int dataCounter);
    public void GradientCostReset();
    public void FullReset();
}