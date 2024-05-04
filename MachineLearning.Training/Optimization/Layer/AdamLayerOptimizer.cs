using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Layer;

public sealed class AdamLayerOptimizer : ILayerOptimizer<Number>
{
    public RecordingLayer Layer { get; }
    public ICostFunction CostFunction { get; }
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


    public AdamLayerOptimizer(AdamOptimizer optimizer, RecordingLayer layer, ICostFunction costFunction)
    {
        Optimizer = optimizer;
        Layer = layer;
        CostFunction = costFunction;

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
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        foreach (int outputNodeIndex in ..Layer.OutputNodeCount)
        {
            FirstMomentBiases[outputNodeIndex] = Optimizer.GradientsDecayRate * FirstMomentBiases[outputNodeIndex] + (1 - Optimizer.GradientsDecayRate) * GradientCostBiases[outputNodeIndex];
            SecondMomentBiases[outputNodeIndex] = Optimizer.SquaredGradientsDecayRate * SecondMomentBiases[outputNodeIndex] + (1 - Optimizer.SquaredGradientsDecayRate) * GradientCostBiases[outputNodeIndex] * GradientCostBiases[outputNodeIndex];
            var mHatB = FirstMomentBiases[outputNodeIndex] / (1 - Math.Pow(Optimizer.GradientsDecayRate, Optimizer.Iteration));
            var vHatB = SecondMomentBiases[outputNodeIndex] / (1 - Math.Pow(Optimizer.SquaredGradientsDecayRate, Optimizer.Iteration));
            Layer.Biases[outputNodeIndex] -= averagedLearningRate * mHatB / (Math.Sqrt(vHatB) + Optimizer.Epsilon);

            foreach (int inputNodeIndex in ..Layer.InputNodeCount)
            {
                FirstMomentWeights[inputNodeIndex, outputNodeIndex] = Optimizer.GradientsDecayRate * FirstMomentWeights[inputNodeIndex, outputNodeIndex] + (1 - Optimizer.GradientsDecayRate) * GradientCostWeights[inputNodeIndex, outputNodeIndex];
                SecondMomentWeights[inputNodeIndex, outputNodeIndex] = Optimizer.SquaredGradientsDecayRate * SecondMomentWeights[inputNodeIndex, outputNodeIndex] + (1 - Optimizer.SquaredGradientsDecayRate) * GradientCostWeights[inputNodeIndex, outputNodeIndex] * GradientCostWeights[inputNodeIndex, outputNodeIndex];
                var mHatW = FirstMomentWeights[inputNodeIndex, outputNodeIndex] / (1 - Math.Pow(Optimizer.GradientsDecayRate, Optimizer.Iteration));
                var vHatW = SecondMomentWeights[inputNodeIndex, outputNodeIndex] / (1 - Math.Pow(Optimizer.SquaredGradientsDecayRate, Optimizer.Iteration));
                Layer.Weights[inputNodeIndex, outputNodeIndex] -= averagedLearningRate * mHatW / (Math.Sqrt(vHatW) + Optimizer.Epsilon);

            }
        }
        /*
        (Number newW, Number newM, Number newV) UpdateWeight(Number currentW, Number lastM, Number lastV, Number gradient){
            var m = Optimizer.GradientsDecayRate*lastM + (1 - Optimizer.GradientsDecayRate)*gradient;
            var v = Optimizer.SquaredGradientsDecayRate*lastV + (1 - Optimizer.SquaredGradientsDecayRate)*gradient*gradient;
            var mHat = m/(1- Math.Pow(Optimizer.GradientsDecayRate, Optimizer.Iteration));
            var vHat = m/(1- Math.Pow(Optimizer.SquaredGradientsDecayRate, Optimizer.Iteration));
            return (currentW-averagedLearningRate*mHat/(Math.Sqrt(vHat)+Optimizer.Epsilon), m, v);
        }
        */
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