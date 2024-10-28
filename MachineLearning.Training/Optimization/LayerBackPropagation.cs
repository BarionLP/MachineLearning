using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public static class LayerBackPropagation
{
    public static Vector ComputeOutputLayerErrors(ILayer layer, ICostFunction costFunction, Vector expected, ILayerSnapshot snapshot) => layer switch
    {
        SimpleLayer simpleLayer => ComputeOutputLayerErrors(simpleLayer, costFunction, expected, (LayerSnapshots.Simple)snapshot),
        _ => throw new NotImplementedException($"Cannot compute output layer errors for {layer}"),
    };

    public static Vector ComputeHiddenLayerErrors(ILayer layer, ILayer nextLayer, Vector nextErrors, ILayerSnapshot snapshot) => (layer, nextLayer) switch
    {
        (SimpleLayer simpleLayer, SimpleLayer simpleNextLayer) => ComputeHiddenLayerErrors(simpleLayer, simpleNextLayer, nextErrors, LayerSnapshots.Is<LayerSnapshots.Simple>(snapshot)),
        (StringEmbeddingLayer stringLayer, SimpleLayer simpleNextLayer) => ComputeHiddenLayerErrors(stringLayer, simpleNextLayer, nextErrors, LayerSnapshots.Is<LayerSnapshots.Embedding>(snapshot)),
        (IEmbedder<string, char>, SimpleLayer) => nextErrors,
        (IEmbedder<float[], int>, SimpleLayer) => nextErrors,
        _ => throw new NotImplementedException($"Cannot compute hidden layer errors for {layer} -> {nextLayer}."),
    };



    public static Vector ComputeHiddenLayerErrors(SimpleLayer layer, SimpleLayer nextLayer, Vector nextErrors, LayerSnapshots.Simple snapshot)
    {
        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var weightedInputDerivatives = nextErrors.Multiply(nextLayer.Weights);
        weightedInputDerivatives.PointwiseMultiplyToSelf(activationDerivatives);

        NumericsDebug.AssertValidNumbers(weightedInputDerivatives);
        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }

    public static Vector ComputeHiddenLayerErrors(StringEmbeddingLayer layer, SimpleLayer nextLayer, Vector nextErrors, LayerSnapshots.Embedding snapshot)
    {
        //activationFunction'(x) = 1
        return nextErrors.Multiply(nextLayer.Weights);
    }

    public static Vector ComputeOutputLayerErrors(SimpleLayer layer, ICostFunction costFunction, Vector expected, LayerSnapshots.Simple snapshot)
    {
        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var costDerivatives = costFunction.Derivative(snapshot.LastActivatedWeights, expected);

        NumericsDebug.AssertValidNumbers(costDerivatives);
        costDerivatives.PointwiseMultiplyToSelf(activationDerivatives);

        return costDerivatives;
    }
}
