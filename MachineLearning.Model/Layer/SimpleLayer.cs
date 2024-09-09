﻿using MachineLearning.Model.Activation;

namespace MachineLearning.Model.Layer;

public sealed class SimpleLayer(Matrix Weights, Vector Biases, IActivationMethod Activation) : ILayer
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix Weights { get; } = Weights; // output * input!!
    public Vector Biases { get; } = Biases;

    public IActivationMethod ActivationFunction { get; } = Activation;

    public uint ParameterCount => (uint)Biases.Count + (uint)Weights.FlatCount;

    public Vector Forward(Vector input)
    {
        var result = Weights.Multiply(input);
        result.AddInPlace(Biases);
        ActivationFunction.Activate(result, result);

        return result;
    }
}
