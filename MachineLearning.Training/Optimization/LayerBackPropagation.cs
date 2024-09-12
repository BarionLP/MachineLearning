using System;
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

    public static Vector ComputeHiddenLayerErrors(ILayer layer, ILayer nextLayer, Vector nextErrors, object snapshot) => (layer, nextLayer) switch
    {
        (SimpleLayer simpleLayer, SimpleLayer simpleNextLayer) => ComputeHiddenLayerErrors(simpleLayer, simpleNextLayer, nextErrors, (LayerSnapshots.Simple)snapshot),
        (StringEmbeddingLayer stringLayer, SimpleLayer simpleNextLayer) => ComputeHiddenLayerErrors(stringLayer, simpleNextLayer, nextErrors, (LayerSnapshots.Embedding)snapshot),
        (IEmbedder<string, char>, SimpleLayer) => nextErrors,
        _ => throw new NotImplementedException($"Cannot compute hidden layer errors for {layer} -> {nextLayer}."),
    };



    public static Vector ComputeHiddenLayerErrors(SimpleLayer layer, SimpleLayer nextLayer, Vector nextErrors, LayerSnapshots.Simple snapshot)
    {

        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var weightedInputDerivatives = nextErrors.Multiply(nextLayer.Weights);
        weightedInputDerivatives.PointwiseMultiplyToSelf(activationDerivatives);

        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }
    
    public static Vector ComputeHiddenLayerErrors(StringEmbeddingLayer layer, SimpleLayer nextLayer, Vector nextErrors, LayerSnapshots.Embedding snapshot)
    {
        var weightedInputDerivatives = nextErrors.Multiply(nextLayer.Weights);
        // weightedInputDerivatives.MultiplyToSelf(1);
        // var activationDerivatives = Vector.Create(layer.OutputNodeCount);

        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }

    public static Vector ComputeOutputLayerErrors(SimpleLayer layer, ICostFunction costFunction, Vector expected, LayerSnapshots.Simple snapshot)
    {
        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var costDerivatives = costFunction.Derivative(snapshot.LastActivatedWeights, expected);

        NumericsDebug.AssertValidNumbers(activationDerivatives);
        NumericsDebug.AssertValidNumbers(costDerivatives);

        costDerivatives.PointwiseMultiplyToSelf(activationDerivatives);

        return costDerivatives;
    }
}