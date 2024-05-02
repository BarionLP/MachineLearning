using Simple.Network.Layer;
using Simple.Training.Cost;
using Simple.Training.Optimization.Layer;

namespace Simple.Training.Optimization;

public interface IOptimizerConfig<TData> {
    public Number CurrentLearningRate { get; }
    public ICostFunction CostFunction { get; }
    public void Init() {}
    public void OnBatchCompleted() {}
    public void OnEpochCompleted() {}
    public ILayerOptimizer<TData> CreateLayerOptimizer(RecordingLayer layer);   
}
