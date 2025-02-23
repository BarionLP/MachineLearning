using System.ComponentModel;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.AdamW;

public sealed class AdamWOptimizer : AdamOptimizer
{
    public new static LayerOptimizerRegistry<AdamWOptimizer> Registry { get; } = [];
    protected override LayerOptimizerRegistry RegistryGetter => Registry;

    public Weight WeightDecayCoefficient /*λ*/ { get; init; } = 1e-3f; // (1e-5 - 1e-2)
    
    static AdamWOptimizer()
    {
        Registry.Register<FeedForwardLayer>((op, layer) => new SimpleAdamWOptimizer(op, layer));
    }
}