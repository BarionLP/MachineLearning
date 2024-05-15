namespace MachineLearning.Domain.Activation;

public sealed class SoftmaxActivation : IActivationMethod<double>
{
    public static readonly SoftmaxActivation Instance = new();

    public double[] Activate(double[] input)
    {
        //var maxInput = input.Max(); // Subtracting max for numerical stability (update derivative!!!)
        var result = new double[input.Length];
        var sum = 0.0;

        foreach (var i in ..input.Length)
        {
            result[i] = Math.Exp(input[i]/* -maxInput */);
            sum += result[i];
        }

        foreach (var i in ..input.Length)
        {
            result[i] /= sum;
        }

        return result;
    }

    // adapted from Sebastian Lague
    public double[] Derivative(double[] input)
    {
        var result = new double[input.Length];

        foreach (var i in ..input.Length)
        {
            result[i] = Math.Exp(input[i]);
        }
        var expSum = result.Sum();

        foreach (var i in ..result.Length)
        {
            var ex = result[i];
            result[i] = (ex * expSum - ex * ex) / (expSum * expSum);
        }

        return result;
    }
}
