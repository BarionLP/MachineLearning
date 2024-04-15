namespace Simple.Training.Cost;

/// <summary>
/// Cross-Entropy Cost Function
/// classification tasks, particularly binary
/// Cons: Numerically unstable (e.g., log(0) issues)
/// </summary>
public sealed class CrossEntropyCost : ICostFunction
{
    public static readonly CrossEntropyCost Instance = new();

    public Number Cost(Number outputActivation, Number expected) =>
        -(expected * Math.Log(outputActivation) + (1 - expected) * Math.Log(1 - outputActivation));

    public Number Derivative(Number outputActivation, Number expected) =>
        (outputActivation - expected) / (outputActivation * (1 - outputActivation));
}