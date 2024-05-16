using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public interface IOptimizer<TWeight> where TWeight : struct, IEquatable<TWeight>, IFormattable
{
    public void Init() { }
    public void OnBatchCompleted() { }
    public void OnEpochCompleted() { }
    public ILayerOptimizer<TWeight> CreateLayerOptimizer(RecordingLayer layer);

}
