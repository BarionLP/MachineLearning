using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public interface IOptimizer<TData>
{
    public Number LearningRate { get; }
    public ICostFunction CostFunction { get; }
    public void Init() { }
    public void OnBatchCompleted() { }
    public void OnEpochCompleted() { }
    public ILayerOptimizer<TData> CreateLayerOptimizer(RecordingLayer layer);
}
