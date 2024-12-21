using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Training.Optimization.Adam;

public class AdamOptimizer : Optimizer
{
    //public required Weight LearningRate { get; init; } = 0.1f;
    public Weight FirstDecayRate { get; init; } = 0.9f;
    public Weight SecondDecayRate { get; init; } = 0.99f; //or 0.999
    public Weight Epsilon { get; init; } = 1e-8f;

    public Weight Iteration { get; set; } = 1; // even when retraining!

    public AdamOptimizer() : base()
    {
        Register<FeedForwardLayer>((layer) => new SimpleAdamOptimizer(this, layer));
    }

    public override void OnBatchCompleted()
    {
        Iteration++;
    }
}