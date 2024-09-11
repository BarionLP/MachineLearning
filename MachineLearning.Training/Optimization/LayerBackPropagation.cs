using System;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public static class LayerBackPropagation
{
    public static Vector ComputeOutputLayerErrors(ILayer layer, ICostFunction costFunction, Vector expected, object snapshot) => layer switch
    {
        SimpleLayer simpleLayer => ComputeOutputLayerErrors(simpleLayer, costFunction, expected, (LayerSnapshot)snapshot),
        _ => throw new NotImplementedException($"Cannot compute output layer errors for {layer}"),
    };

    public static Vector ComputeHiddenLayerErrors(ILayer layer, ILayer nextLayer, Vector nextErrors, object snapshot) => (layer, nextLayer) switch
    {
        (SimpleLayer simpleLayer, SimpleLayer simpleNextLayer) => ComputeHiddenLayerErrors(simpleLayer, simpleNextLayer, nextErrors, (LayerSnapshot)snapshot),
        _ => throw new NotImplementedException($"Cannot compute hidden layer errors for {layer} -> {nextLayer}."),
    };



    public static Vector ComputeHiddenLayerErrors(SimpleLayer layer, SimpleLayer nextLayer, Vector nextErrors, LayerSnapshot snapshot)
    {

        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var weightedInputDerivatives = nextErrors.Multiply(nextLayer.Weights);
        weightedInputDerivatives.PointwiseMultiplyToSelf(activationDerivatives);

        return weightedInputDerivatives; // contains now the error values (weightedInputDerivatives*activationDerivatives)
    }

    public static Vector ComputeOutputLayerErrors(SimpleLayer layer, ICostFunction costFunction, Vector expected, LayerSnapshot snapshot)
    {
        var activationDerivatives = layer.ActivationFunction.Derivative(snapshot.LastWeightedInput);
        var costDerivatives = costFunction.Derivative(snapshot.LastActivatedWeights, expected);

#if DEBUG
        if (activationDerivatives.AsSpan().Contains(double.NaN))
        {
            Console.WriteLine("Model weights contain NaN");
        }
        if (costDerivatives.AsSpan().Contains(double.NaN))
        {
            Console.WriteLine("Model weights contain NaN");
        }
#endif
        costDerivatives.PointwiseMultiplyToSelf(activationDerivatives);

        return costDerivatives;
    }
}
