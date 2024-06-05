using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public interface IOptimizer
{
    public void Init() { }
    public void OnBatchCompleted() { }
    public void OnEpochCompleted() { }
    public ILayerOptimizer CreateLayerOptimizer(RecordingLayer layer);
}
