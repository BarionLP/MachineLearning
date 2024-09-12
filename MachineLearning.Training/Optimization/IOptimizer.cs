using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public interface IGenericOptimizer
{
    public Weight LearningRate { get; }
    public ICostFunction CostFunction { get; }
    public void Init() { }
    public void OnBatchCompleted() { }
    public void OnEpochCompleted() { }
    public ILayerOptimizer CreateLayerOptimizer(ILayer layer);
}
