using Ametrin.Guards;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Training.Optimization;

public interface ILayerOptimizer
{
    public void Update(Vector nodeValues, ILayerSnapshot snapshot, IGradients gradients);
    public void Apply(IGradients gradients);
    public void FullReset();
};

public interface ILayerOptimizer<TLayer, TSnapshot> : ILayerOptimizer where TLayer : ILayer where TSnapshot : ILayerSnapshot
{
    public void Update(Vector nodeValues, TSnapshot snapshot, IGradients gradients);
    void ILayerOptimizer.Update(Vector nodeValues, ILayerSnapshot snapshot, IGradients gradients) => Update(nodeValues, Guard.Is<TSnapshot>(snapshot), gradients);
}
