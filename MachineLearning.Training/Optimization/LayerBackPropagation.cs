using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public static class LayerBackPropagation
{
    public static Vector ComputeHiddenLayerErrors(ILayer layer, ILayer nextLayer, Vector nextErrors, ILayerSnapshot snapshot) => (layer, nextLayer) switch
    {
        (FeedForwardLayer simpleLayer, FeedForwardLayer simpleNextLayer) => ComputeHiddenLayerErrors(simpleLayer, simpleNextLayer, nextErrors, LayerSnapshots.Is<LayerSnapshots.Simple>(snapshot)),
        (StringEmbeddingLayer stringLayer, FeedForwardLayer simpleNextLayer) => ComputeHiddenLayerErrors(stringLayer, simpleNextLayer, nextErrors, LayerSnapshots.Is<LayerSnapshots.Embedding>(snapshot)),
        (IEmbedder<string, char>, FeedForwardLayer) => nextErrors,
        (IEmbedder<double[], int>, FeedForwardLayer) => nextErrors,
        _ => throw new NotImplementedException($"Cannot compute hidden layer errors for {layer} -> {nextLayer}."),
    };



    public static Vector ComputeHiddenLayerErrors(FeedForwardLayer layer, FeedForwardLayer nextLayer, Vector nextErrors, LayerSnapshots.Simple snapshot)
    {
        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var weightedInputDerivatives = nextErrors.Multiply(nextLayer.Weights);
        weightedInputDerivatives.PointwiseMultiplyToSelf(activationDerivatives);
        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }

    public static Vector ComputeHiddenLayerErrors(StringEmbeddingLayer layer, FeedForwardLayer nextLayer, Vector nextErrors, LayerSnapshots.Embedding snapshot)
    {
        //activationFunction'(x) = 1
        return nextErrors.Multiply(nextLayer.Weights);
    }

    public static Vector ComputeOutputLayerErrors(FeedForwardLayer layer, ICostFunction costFunction, Vector expected, LayerSnapshots.Simple snapshot)
    {
        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var costDerivatives = costFunction.Derivative(snapshot.LastActivatedWeights, expected);

        costDerivatives.PointwiseMultiplyToSelf(activationDerivatives);

        return costDerivatives;
    }
}
