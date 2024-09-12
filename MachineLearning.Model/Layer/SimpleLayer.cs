﻿using System.Diagnostics;
using MachineLearning.Model.Activation;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public sealed class SimpleLayer(Matrix Weights, Vector Biases, IActivationFunction Activation) : ILayer
{
    public int InputNodeCount { get; } = Weights.ColumnCount;
    public int OutputNodeCount { get; } = Biases.Count;
    public Matrix Weights { get; } = Weights; // output * input!!
    public Vector Biases { get; } = Biases;

    public IActivationFunction ActivationFunction { get; } = Activation;

    public uint ParameterCount => (uint)Biases.Count + (uint)Weights.FlatCount;

    public Vector Forward(Vector input)
    {
        var result = Weights.Multiply(input);
        result.AddToSelf(Biases);
        ActivationFunction.ActivateTo(result, result);

        return result;
    }

    public Vector Forward(Vector input, ILayerSnapshot snapshot)
    {
        if (snapshot is not LayerSnapshots.Simple simpleSnapshot) throw new UnreachableException();
        input.CopyTo(simpleSnapshot.LastRawInput);
        Weights.MultiplyTo(input, simpleSnapshot.LastWeightedInput);
        simpleSnapshot.LastWeightedInput.AddToSelf(Biases);

        ActivationFunction.ActivateTo(simpleSnapshot.LastWeightedInput, simpleSnapshot.LastActivatedWeights);

        return simpleSnapshot.LastActivatedWeights;
    }
}
