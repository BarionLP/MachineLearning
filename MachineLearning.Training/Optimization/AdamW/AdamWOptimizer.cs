﻿using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.AdamW;

public sealed class AdamWOptimizer : AdamOptimizer
{
    public Weight WeightDecayCoefficient /*λ*/ { get; init; } = 1e-3f; // (1e-5 - 1e-2)
 
    public AdamWOptimizer() : base()
    {
        Register<FeedForwardLayer>((layer) => new SimpleAdamWOptimizer(this, layer));
    }
}