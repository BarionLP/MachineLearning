namespace ML.Core.Evaluation.Cost;

/// <summary>
/// Cross-Entropy Cost Function<br/>
/// requires outputs in range 0..1<br/>
/// prefer <see cref="CrossEntropyCostFromLogits"/>
/// Cons: Numerically unstable (e.g., log(0) issues), this impl clamps to <see cref="EPSILON"/><br/>
/// </summary>
public sealed class CrossEntropyCostFromProbabilities : ICostFunction
{
    public static CrossEntropyCostFromProbabilities Instance => field ??= new();
    const Weight EPSILON = 1e-7f;

    public Weight Cost(Weight output, Weight expected)
    {
        output = Weight.Clamp(output, EPSILON, 1 - EPSILON);
        return -expected * Weight.Log(output);
    }

    public Weight Derivative(Weight output, Weight expected)
    {
        return -expected / Weight.Clamp(output, EPSILON, 1 - EPSILON);
        // return output - expected;
    }
}
