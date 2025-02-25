using System.IO;
using MachineLearning.Model.Activation;
using MachineLearning.Model.Attributes;
using MachineLearning.Model.Layer;
using MachineLearning.Serialization;
using MachineLearning.Training.Attributes;

namespace ML.MultiLayerPerceptron;

[GeneratedLayer, GenerateOptimizers]
public sealed partial class PerceptronLayer : ILayer<Vector, PerceptronLayer.Snapshot>
{
    public int InputNodeCount => Weights.ColumnCount;
    public int OutputNodeCount => Biases.Count;
    [Weights] public Matrix Weights { get; } // output * input!!
    [Weights] public Vector Biases { get; }

    [Parameter] public IActivationFunction ActivationFunction { get; }

    public Vector Forward(Vector input)
    {
        var result = Weights.Multiply(input);
        result.AddToSelf(Biases);
        ActivationFunction.ActivateTo(result, result);

        return result;
    }

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        input.CopyTo(snapshot.LastRawInput);
        Weights.MultiplyTo(input, snapshot.LastWeightedInput);
        snapshot.LastWeightedInput.AddToSelf(Biases);

        ActivationFunction.ActivateTo(snapshot.LastWeightedInput, snapshot.LastActivatedWeights);

        return snapshot.LastActivatedWeights;
    }

    public void Backward(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        VectorHelper.MultiplyToMatrixAddTo(outputGradient, snapshot.LastRawInput, gradients.Weights); // GradientCostWeights.AddInPlaceMultiplied ?
    }

    partial class Snapshot
    {
        public Vector LastRawInput { get; } = Vector.Create(layer.InputNodeCount);
        public Vector LastWeightedInput { get; } = Vector.Create(layer.OutputNodeCount);
        public Vector LastActivatedWeights { get; } = Vector.Create(layer.OutputNodeCount);
    }

    public static Result<PerceptronLayer> ReadLegacy(BinaryReader reader)
    {
        var inputNodeCount = reader.ReadInt32();
        var outputNodeCount = reader.ReadInt32();
        var activationMethod = ActivationFunctionSerializer.Read(reader);
        var layerBuilder = new LayerFactory(inputNodeCount, outputNodeCount).SetActivationFunction(activationMethod);
        var layer = layerBuilder.Create();

        // decode weights & biases
        foreach (var outputIndex in ..outputNodeCount)
        {
            layer.Biases[outputIndex] = reader.ReadSingle();
            foreach (var inputIndex in ..inputNodeCount)
            {
                layer.Weights[outputIndex, inputIndex] = reader.ReadSingle();
            }
        }

        return layer;
    }
}
