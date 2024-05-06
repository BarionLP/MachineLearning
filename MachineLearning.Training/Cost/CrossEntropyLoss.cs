namespace MachineLearning.Training.Cost;

/// <summary>
/// Cross-Entropy Cost Function<br/>
/// classification tasks, particularly binary<br/>
/// Cons: Numerically unstable (e.g., log(0) issues)<br/>
/// </summary>
public sealed class CrossEntropyLoss : ICostFunction
{
    public static readonly CrossEntropyLoss Instance = new();

    public Number Cost(Number outputActivation, Number expected) =>
        -(expected * Math.Log(outputActivation) + (1 - expected) * Math.Log(1 - outputActivation));

    public Number Derivative(Number outputActivation, Number expected) =>
        (outputActivation - expected) / (outputActivation * (1 - outputActivation));
}
