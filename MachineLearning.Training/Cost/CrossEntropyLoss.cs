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
    const double EPSILON = 1e-10;

    public double Cost(double outputActivation, double expected)
    {
        outputActivation = Math.Clamp(outputActivation, EPSILON, 1 - EPSILON); //just return 0 or 1?
        return -(expected * Math.Log(outputActivation) + (1 - expected) * Math.Log(1 - outputActivation));
    }

    public double Derivative(double outputActivation, double expected)
    {
        outputActivation = Math.Clamp(outputActivation, EPSILON, 1 - EPSILON); //just return 0 or 1?
        return (outputActivation - expected) / (outputActivation * (1 - outputActivation));
    }
}
