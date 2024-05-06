namespace MachineLearning.Training.Cost;


public class CategoricalCrossEntropy : ICostFunction
{
    public static readonly CategoricalCrossEntropy Instance = new();
    public double Cost(double output, double expected)
    {
        if (expected == 0)
        {
            return -Math.Log(1 - output);
        }
        else
        {
            return -Math.Log(output);
        }
    }

    public double Derivative(double output, double expected)
    {
        if (output == 0 || output == 1)
        {
            throw new InvalidOperationException("Derivative is not defined for output values 0 or 1.");
        }

        return -(expected / output) + ((1 - expected) / (1 - output));
    }
}

/*
/// <summary>
/// Good for classification
/// only when expected is 0 or 1
/// </summary>
public sealed class CategoricalCrossEntropy(double Epsilon) : ICostFunction {
    public static readonly CategoricalCrossEntropy Instance = new(1e-15);

    public double Epsilon { get; } = Epsilon;

    public double Cost(double output, double expected)
    {
        return -expected * Math.Log(output + Epsilon);
    }

    public double Derivative(double output, double expected)
    {
        return -expected / (output + Epsilon);
    }
*/