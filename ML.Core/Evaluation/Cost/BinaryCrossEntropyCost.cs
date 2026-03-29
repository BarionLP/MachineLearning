namespace ML.Core.Evaluation.Cost;

/// <summary>
/// Binary-Cross-Entropy Cost Function<br/>
/// classification tasks, particularly binary<br/>
/// requires outputs in range 0..1<br/>
/// Cons: Numerically unstable (e.g., log(0) issues), this impl clamps to <see cref="EPSILON"/><br/>
/// </summary>
// TODO: FromLogits version see CrossEntropyCostFromLogits but with Sigmoid
public sealed class BinaryCrossEntropyCost : ICostFunction
{
    public static BinaryCrossEntropyCost Instance => field ??= new();
    public const Weight EPSILON = 1e-7f;

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