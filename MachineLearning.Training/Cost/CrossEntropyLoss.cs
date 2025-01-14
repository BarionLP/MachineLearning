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

    public Weight Cost(Weight output, Weight expected)
    {
        output = float.Clamp(output, EPSILON, 1 - EPSILON); //just return 0 or 1?
        return -(expected * float.Log(output) + (1 - expected) * float.Log(1 - output));
    }

    public Weight Derivative(Weight output, Weight expected)
    {
        output = float.Clamp(output, EPSILON, 1 - EPSILON); //just return 0 or 1?
        return (output - expected) / (output * (1 - output));
    }
}
