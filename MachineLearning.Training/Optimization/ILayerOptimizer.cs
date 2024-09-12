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


public interface ILayerOptimizer<TLayer, TSnapshot> : ILayerOptimizer where TLayer : ILayer where TSnapshot : ILayerSnapshot
{
    public new TLayer Layer { get; }
    ILayer ILayerOptimizer.Layer => Layer;
    public void Update(Vector nodeValues, TSnapshot snapshot);
    void ILayerOptimizer.Update(Vector nodeValues, ILayerSnapshot snapshot) => Update(nodeValues, LayerSnapshots.Is<TSnapshot>(snapshot));
}