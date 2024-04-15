﻿namespace Simple.Training.Cost;

/// <summary>
/// Mean Squared Error (MSE) Cost Function <br/>
/// widely used for regression <br/>
/// Cons: Can be sensitive to outliers, often bad for classification <br/>
/// </summary>
public sealed class MeanSquaredErrorCost : ICostFunction
{
    public static readonly MeanSquaredErrorCost Instance = new();
    public Number Cost(Number outputActivation, Number expected) =>
        0.5 * Math.Pow(outputActivation - expected, 2);

    public Number Derivative(Number outputActivation, Number expected) =>
        outputActivation - expected;
}