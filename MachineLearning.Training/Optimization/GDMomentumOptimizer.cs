using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public sealed class GDMomentumOptimizer(GDMomentumOptimizerConfig config) : IOptimizer<double>
{
    public GDMomentumOptimizerConfig Config { get; } = config;
    public double LearningRate { get; private set; }

    public void Init()
    {
        LearningRate = Config.LearningRate;
    }

    public void OnEpochCompleted()
    {
        LearningRate *= Config.LearningRateEpochMultiplier;
    }
    public ILayerOptimizer<double> CreateLayerOptimizer(RecordingLayer layer) => new GDMomentumLayerOptimizer(this, layer);
}