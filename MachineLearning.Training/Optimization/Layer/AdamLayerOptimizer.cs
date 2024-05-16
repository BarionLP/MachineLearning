using MachineLearning.Domain;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Layer;

public sealed class AdamLayerOptimizer : ILayerOptimizer<double>
{
    public RecordingLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.Config.CostFunction;
    public AdamOptimizer Optimizer { get; }

    public readonly Vector<double> GradientCostBiases;
    public readonly Matrix<double> GradientCostWeights;

    // formula symbol M 
    // exponentially decaying average of past gradients. It is akin to the mean of the gradients.
    public readonly Vector<double> FirstMomentBiases;
    public readonly Matrix<double> FirstMomentWeights;

    // formula symbol V
    // exponentially decaying average of the squared gradients. It is akin to the uncentered variance of the gradients.
    public readonly Vector<double> SecondMomentBiases;
    public readonly Matrix<double> SecondMomentWeights;


    public AdamLayerOptimizer(AdamOptimizer optimizer, RecordingLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientCostBiases = Vector.Build.Dense(Layer.OutputNodeCount);
        GradientCostWeights = Matrix.Build.Dense(Layer.OutputNodeCount, Layer.InputNodeCount);

        FirstMomentBiases = Vector.Build.Dense(Layer.OutputNodeCount);
        FirstMomentWeights = Matrix.Build.Dense(Layer.OutputNodeCount, Layer.InputNodeCount);

        SecondMomentBiases = Vector.Build.Dense(Layer.OutputNodeCount);
        SecondMomentWeights = Matrix.Build.Dense(Layer.OutputNodeCount, Layer.InputNodeCount);
    }

    public void Update(Vector<double> nodeValues)
    {
        // Compute the gradient for weights
        var inputTranspose = Layer.LastRawInput.ToRowMatrix();
        var nodeValuesTranspose = nodeValues.ToColumnMatrix();
        var weightGradients = nodeValuesTranspose * inputTranspose;

        // Update GradientCostWeights
        GradientCostWeights.AddInPlace(weightGradients);

        // Update GradientCostBiases
        GradientCostBiases.AddInPlace(nodeValues);

        // foreach (int outputNodeIndex in ..Layer.OutputNodeCount)
        // {
        //     foreach (int inputNodeIndex in ..Layer.InputNodeCount)
        //     {
        //         // partial derivative cost with respect to weight of current connection
        //         var derivativeCostWrtWeight = Layer.LastRawInput[inputNodeIndex] * nodeValues[outputNodeIndex];
        //         GradientCostWeights[inputNodeIndex, outputNodeIndex] += derivativeCostWrtWeight;
        //     }

        //     // derivative cost with respect to bias (bias' = 1)
        //     var derivativeCostWrtBias = 1 * nodeValues[outputNodeIndex];
        //     GradientCostBiases[outputNodeIndex] += derivativeCostWrtBias;
        // }
    }

    public void Apply(int dataCounter)
    {
        // do i need gradient clipping?
        var averagedLearningRate = Optimizer.Config.LearningRate / Math.Sqrt(dataCounter);

        // parallelizing makes no difference
        // Update biases
        FirstMomentBiases.MapIndexedInplace((i, fm) => FirstMomentEstimate(fm, GradientCostBiases[i]));
        SecondMomentBiases.MapIndexedInplace((i, sm) => SecondMomentEstimate(sm, GradientCostBiases[i]));
        Layer.Biases.SubtractInPlace(FirstMomentBiases.MapIndexed((i, fm) => WeightReduction(fm, SecondMomentBiases[i])));

        // Update weights
        FirstMomentWeights.MapIndexedInplace((i, j, fm) => FirstMomentEstimate(fm, GradientCostWeights[i, j]));
        SecondMomentWeights.MapIndexedInplace((i, j, sm) => SecondMomentEstimate(sm, GradientCostWeights[i, j]));
        Layer.Weights.SubtractInPlace(FirstMomentWeights.MapIndexed((i, j, fm) => WeightReduction(fm, SecondMomentWeights[i, j])));

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
        GradientCostBiases.Clear();
        GradientCostWeights.Clear();
    }

    public void FullReset()
    {
        GradientCostBiases.Clear();
        FirstMomentBiases.Clear();
        SecondMomentBiases.Clear();

        GradientCostWeights.Clear();
        FirstMomentWeights.Clear();
        SecondMomentWeights.Clear();
    }
}