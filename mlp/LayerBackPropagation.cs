using Ametrin.Guards;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace ML.MultiLayerPerceptron;

public static class LayerBackPropagation
{
    // public static Vector ComputeHiddenLayerErrors(ILayer layer, ILayer nextLayer, Vector nextErrors, ILayerSnapshot snapshot) => (layer, nextLayer) switch
    // {
    //     (PerceptronLayer simpleLayer, PerceptronLayer simpleNextLayer) => ComputeHiddenLayerErrors(simpleLayer, simpleNextLayer, nextErrors, Guard.Is<PerceptronLayer.Snapshot>(snapshot)),
    //     //(TrainedEmbeddingLayer stringLayer, FeedForwardLayer simpleNextLayer) => ComputeHiddenLayerErrors(stringLayer, simpleNextLayer, nextErrors, LayerSnapshots.Is<LayerSnapshots.Embedding>(snapshot)),
    //     // (IEmbedder<string, char>, FeedForwardLayer) => nextErrors,
    //     // (IEmbedder<double[], int>, FeedForwardLayer) => nextErrors,
    //     _ => throw new NotImplementedException($"Cannot compute hidden layer errors for {layer} -> {nextLayer}."),
    // };


    // public static Vector ComputeHiddenLayerErrors(PerceptronLayer layer, PerceptronLayer nextLayer, Vector nextErrors, PerceptronLayer.Snapshot snapshot)
    // {
    //     var weightedGradient = nextErrors.Multiply(nextLayer.Weights);
    //     var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
    //     weightedGradient.PointwiseMultiplyToSelf(activationDerivatives);
    //     return weightedGradient; // contains now the error values (weightedGradient*activationDerivatives)
    // }

    //public static Vector ComputeHiddenLayerErrors(TrainedEmbeddingLayer layer, FeedForwardLayer nextLayer, Vector nextErrors, LayerSnapshots.Embedding snapshot)
    //{
    //    //activationFunction'(x) = 1
    //    return nextErrors.Multiply(nextLayer.Weights);
    //}

    // public static Vector ComputeOutputLayerErrors(PerceptronLayer layer, ICostFunction costFunction, Vector expected, PerceptronLayer.Snapshot snapshot)
    // {
    //     var costGradient = costFunction.Derivative(snapshot.LastActivatedWeights, expected);
    //     var activationGradient = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);

    //     costGradient.PointwiseMultiplyToSelf(activationGradient);

    //     return costGradient;
    // }
}
