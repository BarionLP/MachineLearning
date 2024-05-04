using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public sealed class GDMomentumOptimizer : IOptimizer<double>
{
    public required double InitalLearningRate { get; init; }
    public double LearningRateEpochMultiplier { get; init; } = 1;
    public required double Momentum { get; init; }
    public required double Regularization { get; init; } = 0.99;
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public double LearningRate { get; private set; }

    public void Init()
    {
        LearningRate = InitalLearningRate;
    }

    public void OnEpochCompleted()
    {
        LearningRate *= LearningRateEpochMultiplier;
    }

    public ILayerOptimizer<double> CreateLayerOptimizer(RecordingLayer layer)
    {
        return new GDMomentumLayerOptimizer(this, layer, CostFunction);
    }
}
