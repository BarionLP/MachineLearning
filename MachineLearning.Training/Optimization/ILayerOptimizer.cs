using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public interface ILayerOptimizer
{
    public ILayer Layer { get; }
    public ICostFunction CostFunction { get; }
    public void Update(Vector nodeValues, ILayerSnapshot snapshot);
    public void Apply(int dataCounter);
    public void GradientCostReset();
    public void FullReset();
};


public interface ILayerOptimizer<TLayer> : ILayerOptimizer where TLayer : ILayer
{
    public new TLayer Layer { get; }
    ILayer ILayerOptimizer.Layer => Layer;

}