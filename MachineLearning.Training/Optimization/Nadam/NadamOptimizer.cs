using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.Nadam;

public sealed class NadamOptimizer : AdamOptimizer
{
    public new static LayerOptimizerRegistry<NadamOptimizer> Registry { get; } = [];
    protected override LayerOptimizerRegistry RegistryGetter => Registry;
    static NadamOptimizer()
    {
        Registry.Register<FeedForwardLayer>((op, layer) => new SimpleNadamOptimizer(op, layer));
    }
}
