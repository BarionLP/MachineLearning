using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public interface IOptimizer<TData> 
{
    public void Init() { }
    public void OnBatchCompleted() { }
    public void OnEpochCompleted() { }
    public ILayerOptimizer<TData> CreateLayerOptimizer(RecordingLayer layer);

}
