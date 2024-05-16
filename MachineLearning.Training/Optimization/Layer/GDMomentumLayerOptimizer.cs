/* using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Layer;

public sealed class GDMomentumLayerOptimizer : ILayerOptimizer<Number>
{
    public RecordingLayer Layer { get; }
    public readonly Number[,] CostGradientWeights;
    public readonly Number[] CostGradientBiases;
    public readonly Number[,] WeightVelocities;
    public readonly Number[] BiasVelocities;
    public ICostFunction CostFunction => Optimizer.Config.CostFunction;
    public GDMomentumOptimizer Optimizer { get; }

    public GDMomentumLayerOptimizer(GDMomentumOptimizer optimizer, RecordingLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;
        CostGradientWeights = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];
        CostGradientBiases = new Number[Layer.OutputNodeCount];
        WeightVelocities = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];
        BiasVelocities = new Number[Layer.OutputNodeCount];
    }

    public void Update(Number[] nodeValues)
    {
        foreach (int outputNodeIndex in ..Layer.OutputNodeCount)
        {
            foreach (int inputNodeIndex in ..Layer.InputNodeCount)
            {
                // partial derivative cost with respect to weight of current connection
                var derivativeCostWrtWeight = Layer.LastRawInput[inputNodeIndex] * nodeValues[outputNodeIndex];
                CostGradientWeights[inputNodeIndex, outputNodeIndex] += derivativeCostWrtWeight;
            }

            // derivative cost with respect to bias (bias' = 1)
            var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
            CostGradientBiases[outputNodeIndex] += derivativeCostWrtBias;
        }
    }

    public void Apply(int dataCounter)
    {
        var averagedLearningRate = Optimizer.LearningRate / dataCounter;
        var weightDecay = 1 - Optimizer.Config.Regularization * averagedLearningRate; //used against overfitting

        foreach (int outputNodeIndex in ..Layer.OutputNodeCount)
        {
            var biasVelocity = BiasVelocities[outputNodeIndex] * Optimizer.Config.Momentum - CostGradientBiases[outputNodeIndex] * averagedLearningRate;
            BiasVelocities[outputNodeIndex] = biasVelocity;
            Layer.Biases[outputNodeIndex] += biasVelocity;
            CostGradientBiases[outputNodeIndex] = 0;

            foreach (int inputNodeIndex in ..Layer.InputNodeCount)
            {
                var weight = Layer.Weights[inputNodeIndex, outputNodeIndex];
                var weightVelocity = WeightVelocities[inputNodeIndex, outputNodeIndex] * Optimizer.Config.Momentum - CostGradientWeights[inputNodeIndex, outputNodeIndex] * averagedLearningRate;
                WeightVelocities[inputNodeIndex, outputNodeIndex] = weightVelocity;
                Layer.Weights[inputNodeIndex, outputNodeIndex] = weight * weightDecay + weightVelocity;
            }
        }
    }

    public void GradientCostReset()
    {
        foreach (var outputNodeIndex in ..Layer.OutputNodeCount)
        {
            CostGradientBiases[outputNodeIndex] = 0;
            foreach (var inputNodeIndex in ..Layer.InputNodeCount)
            {
                CostGradientWeights[inputNodeIndex, outputNodeIndex] = 0;
            }
        }
    }

    public void FullReset()
    {
        foreach (var outputNodeIndex in ..Layer.OutputNodeCount)
        {
            CostGradientBiases[outputNodeIndex] = 0;
            BiasVelocities[outputNodeIndex] = 0;
            foreach (var inputNodeIndex in ..Layer.InputNodeCount)
            {
                CostGradientWeights[inputNodeIndex, outputNodeIndex] = 0;
                WeightVelocities[inputNodeIndex, outputNodeIndex] = 0;
            }
        }
    }
} */