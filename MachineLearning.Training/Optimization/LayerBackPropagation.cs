using Ametrin.Guards;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public static class LayerBackPropagation
{
    public static Vector ComputeHiddenLayerErrors(ILayer layer, ILayer nextLayer, Vector nextErrors, ILayerSnapshot snapshot) => (layer, nextLayer) switch
    {
        (FeedForwardLayer simpleLayer, FeedForwardLayer simpleNextLayer) => ComputeHiddenLayerErrors(simpleLayer, simpleNextLayer, nextErrors, Guard.Is<LayerSnapshots.Simple>(snapshot)),
        //(TrainedEmbeddingLayer stringLayer, FeedForwardLayer simpleNextLayer) => ComputeHiddenLayerErrors(stringLayer, simpleNextLayer, nextErrors, LayerSnapshots.Is<LayerSnapshots.Embedding>(snapshot)),
        // (IEmbedder<string, char>, FeedForwardLayer) => nextErrors,
        // (IEmbedder<double[], int>, FeedForwardLayer) => nextErrors,
        _ => throw new NotImplementedException($"Cannot compute hidden layer errors for {layer} -> {nextLayer}."),
    };


    public static Vector ComputeHiddenLayerErrors(FeedForwardLayer layer, FeedForwardLayer nextLayer, Vector nextErrors, LayerSnapshots.Simple snapshot)
    {
        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var weightedGradient = nextErrors.Multiply(nextLayer.Weights);
        weightedGradient.PointwiseMultiplyToSelf(activationDerivatives);
        return weightedGradient; // contains now the error values (weightedGradient*activationDerivatives)
    }

    //public static Vector ComputeHiddenLayerErrors(TrainedEmbeddingLayer layer, FeedForwardLayer nextLayer, Vector nextErrors, LayerSnapshots.Embedding snapshot)
    //{
    //    //activationFunction'(x) = 1
    //    return nextErrors.Multiply(nextLayer.Weights);
    //}

    public static Vector ComputeOutputLayerErrors(FeedForwardLayer layer, ICostFunction costFunction, Vector expected, LayerSnapshots.Simple snapshot)
    {
        var costGradient = costFunction.Derivative(snapshot.LastActivatedWeights, expected);
        var activationGradient = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);

        costGradient.PointwiseMultiplyToSelf(activationGradient);

        return costGradient;
    }
}
