namespace MachineLearning.Training.Cost;

/// <summary>
/// Mean Squared Error (MSE) Cost Function <br/>
/// widely used for regression <br/>
/// Cons: Can be sensitive to outliers, often bad for classification <br/>
/// </summary>
public sealed class MeanSquaredErrorCost : ICostFunction
{
    public static readonly MeanSquaredErrorCost Instance = new();
    public Weight Cost(Weight outputActivation, Weight expected) =>
        0.5f * MathF.Pow(outputActivation - expected, 2);

    public Weight Derivative(Weight outputActivation, Weight expected) =>
        outputActivation - expected;
}
