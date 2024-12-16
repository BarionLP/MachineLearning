namespace MachineLearning.Training.Cost;

/// <summary>
/// Cross-Entropy Cost Function<br/>
/// classification tasks, particularly binary<br/>
/// requires outputs in range 0..1<br/>
/// Cons: Numerically unstable (e.g., log(0) issues)<br/>
/// </summary>
public sealed class CrossEntropyLoss : ICostFunction
{
    public static readonly CrossEntropyLoss Instance = new();
    const Weight EPSILON = 1e-7f;

    public Weight Cost(Weight outputActivation, Weight expected)
    {
        outputActivation = Math.Clamp(outputActivation, EPSILON, 1 - EPSILON); //just return 0 or 1?
        return -(expected * MathF.Log(outputActivation) + (1 - expected) * MathF.Log(1 - outputActivation));
    }

    public Weight Derivative(Weight outputActivation, Weight expected)
    {
        outputActivation = Math.Clamp(outputActivation, EPSILON, 1 - EPSILON); //just return 0 or 1?
        return (outputActivation - expected) / (outputActivation * (1 - outputActivation));
    }
}
