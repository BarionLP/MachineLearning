using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.Nadam;

public sealed class NadamOptimizer : AdamOptimizer
{
    public NadamOptimizer() : base()
    {
        Register<FeedForwardLayer>((layer) => new SimpleNadamOptimizer(this, layer));
    }
}
