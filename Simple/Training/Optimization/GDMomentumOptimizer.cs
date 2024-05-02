using Simple.Network.Layer;
using Simple.Training.Cost;
using Simple.Training.Optimization.Layer;

namespace Simple.Training.Optimization;

public sealed class GDMomentumOptimizer : IOptimizerConfig<double> {
    public required double LearningRate { get; init; }
    public double LearningRateEpochMultiplier { get; init; } = 1;
    public required double Momentum { get; init; }
    public required double Regularization { get; init; } = 0.99;
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public double CurrentLearningRate { get; private set; }

    public void Init() {
        CurrentLearningRate = LearningRate;
    }

    public void OnEpochCompleted(){
        CurrentLearningRate *= LearningRateEpochMultiplier;
    }

    public ILayerOptimizer<double> CreateLayerOptimizer(RecordingLayer layer) {
        return new GDMomentumLayerOptimizer(layer, CostFunction);
    }
}
