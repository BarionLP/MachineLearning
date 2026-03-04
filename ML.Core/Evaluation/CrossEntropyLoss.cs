namespace ML.Core.Evaluation;

/// <summary>
/// Cross-Entropy Cost Function<br/>
/// classification tasks, particularly binary<br/>
/// requires outputs in range 0..1<br/>
/// Cons: Numerically unstable (e.g., log(0) issues)<br/>
/// </summary>
public sealed class CrossEntropyLoss : ICostFunction
{
    public static CrossEntropyLoss Instance => field ??= new();
    const Weight EPSILON = 1e-7f;

    public Weight Cost(Weight output, Weight expected)
    {
        output = Weight.Clamp(output, EPSILON, 1 - EPSILON); // just return 0 or 1?
        return -(expected * Weight.Log(output) + (1 - expected) * Weight.Log(1 - output));
    }

    public Weight Derivative(Weight output, Weight expected)
    {
        output = Weight.Clamp(output, EPSILON, 1 - EPSILON); // just return 0 or 1?
        return (output - expected) / (output * (1 - output));
    }
}

public sealed class CrossEntropyFromSoftmaxLoss : ICostFunction
{
    public static readonly CrossEntropyFromSoftmaxLoss Instance = new();
    const Weight EPSILON = 1e-7f;

    public Weight Cost(Weight output, Weight expected)
    {
        output = Weight.Clamp(output, EPSILON, 1 - EPSILON);
        return -expected * Weight.Log(output);
    }

    public Weight Derivative(Weight output, Weight expected)
    {
        return output - expected;
    }
}
