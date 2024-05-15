using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Layer;

public sealed class AdamLayerOptimizer : ILayerOptimizer<Number>
{
    public RecordingLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.Config.CostFunction;
    public AdamOptimizer Optimizer { get; }

    public readonly Number[] GradientCostBiases;
    public readonly Number[,] GradientCostWeights;

    // formula symbol M 
    // exponentially decaying average of past gradients. It is akin to the mean of the gradients.
    public readonly Number[] FirstMomentBiases;
    public readonly Number[,] FirstMomentWeights;

    // formula symbol V
    // exponentially decaying average of the squared gradients. It is akin to the uncentered variance of the gradients.
    public readonly Number[] SecondMomentBiases;
    public readonly Number[,] SecondMomentWeights;


    public AdamLayerOptimizer(AdamOptimizer optimizer, RecordingLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientCostBiases = new Number[Layer.OutputNodeCount];
        GradientCostWeights = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];

        FirstMomentBiases = new Number[Layer.OutputNodeCount];
        FirstMomentWeights = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];

        SecondMomentBiases = new Number[Layer.OutputNodeCount];
        SecondMomentWeights = new Number[Layer.InputNodeCount, Layer.OutputNodeCount];
    }

    public void Update(double[] nodeValues)
    {
        foreach (int outputNodeIndex in ..Layer.OutputNodeCount)
        {
            foreach (int inputNodeIndex in ..Layer.InputNodeCount)
            {
                // partial derivative cost with respect to weight of current connection
                var derivativeCostWrtWeight = Layer.LastRawInput[inputNodeIndex] * nodeValues[outputNodeIndex];
                GradientCostWeights[inputNodeIndex, outputNodeIndex] += derivativeCostWrtWeight;
            }

            // derivative cost with respect to bias (bias' = 1)
            var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
            GradientCostBiases[outputNodeIndex] += derivativeCostWrtBias;
        }
    }

    public void Apply(int dataCounter)
    {
        // do i need gradient clipping?
        var averagedLearningRate = Optimizer.Config.LearningRate / Math.Sqrt(dataCounter);

        foreach (int outputNodeIndex in ..Layer.OutputNodeCount)
        {
            FirstMomentBiases[outputNodeIndex] = FirstMomentEstimate(FirstMomentBiases[outputNodeIndex], GradientCostBiases[outputNodeIndex]);
            SecondMomentBiases[outputNodeIndex] = SecondMomentEstimate(SecondMomentBiases[outputNodeIndex], GradientCostBiases[outputNodeIndex]);
            Layer.Biases[outputNodeIndex] -= WeightReduction(FirstMomentBiases[outputNodeIndex], SecondMomentBiases[outputNodeIndex]);

            foreach (int inputNodeIndex in ..Layer.InputNodeCount)
            {
                FirstMomentWeights[inputNodeIndex, outputNodeIndex] = FirstMomentEstimate(FirstMomentWeights[inputNodeIndex, outputNodeIndex], GradientCostWeights[inputNodeIndex, outputNodeIndex]);
                SecondMomentWeights[inputNodeIndex, outputNodeIndex] = SecondMomentEstimate(SecondMomentWeights[inputNodeIndex, outputNodeIndex], GradientCostWeights[inputNodeIndex, outputNodeIndex]);
                Layer.Weights[inputNodeIndex, outputNodeIndex] -= WeightReduction(FirstMomentWeights[inputNodeIndex, outputNodeIndex], SecondMomentWeights[inputNodeIndex, outputNodeIndex]);

            }
        }
        
        double WeightReduction(double firstMoment, double secondMoment){
            var mHat = firstMoment / (1 - Math.Pow(Optimizer.Config.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - Math.Pow(Optimizer.Config.SecondDecayRate, Optimizer.Iteration));
            return averagedLearningRate * mHat / (Math.Sqrt(vHat) + Optimizer.Config.Epsilon);
        }
        double FirstMomentEstimate(double lastMoment, double gradient){
            return Optimizer.Config.FirstDecayRate * lastMoment + (1 - Optimizer.Config.FirstDecayRate) * gradient;
        }
        double SecondMomentEstimate(double lastMoment, double gradient){
            return Optimizer.Config.SecondDecayRate * lastMoment + (1 - Optimizer.Config.SecondDecayRate) * gradient*gradient;
        }
    }

    public void GradientCostReset()
    {
        foreach (var outputNodeIndex in ..Layer.OutputNodeCount)
        {
            GradientCostBiases[outputNodeIndex] = 0;
            foreach (var inputNodeIndex in ..Layer.InputNodeCount)
            {
                GradientCostWeights[inputNodeIndex, outputNodeIndex] = 0;
            }
        }
    }

    public void FullReset()
    {
        foreach (var outputNodeIndex in ..Layer.OutputNodeCount)
        {
            GradientCostBiases[outputNodeIndex] = 0;
            FirstMomentBiases[outputNodeIndex] = 0;
            SecondMomentBiases[outputNodeIndex] = 0;
            foreach (var inputNodeIndex in ..Layer.InputNodeCount)
            {
                GradientCostWeights[inputNodeIndex, outputNodeIndex] = 0;
                FirstMomentWeights[inputNodeIndex, outputNodeIndex] = 0;
                SecondMomentWeights[inputNodeIndex, outputNodeIndex] = 0;
            }
        }
    }
}