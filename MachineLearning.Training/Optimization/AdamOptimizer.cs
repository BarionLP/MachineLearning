using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public sealed class AdamOptimizer(AdamOptimizerConfig config) : IOptimizer
{
    public double Iteration { get; set; } = 1; //(even when retraining!) when starting with 0 gradient estimates shoot to infinity?
    public AdamOptimizerConfig Config { get; } = config;

    public void OnBatchCompleted()
    {
        Iteration++;
    }
    public ILayerOptimizer CreateLayerOptimizer(RecordingLayer layer) => new AdamLayerOptimizer(this, layer);
}