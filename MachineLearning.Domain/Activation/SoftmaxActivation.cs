namespace MachineLearning.Domain.Activation;

public sealed class SoftmaxActivation : IActivationMethod<double>
{
    public static readonly SoftmaxActivation Instance = new();

    public Vector<double> Activate(Vector<double> input)
    {
        var result = input.PointwiseExp();
        var sum = result.Sum();
        return result / sum;
    }

    // adapted from Sebastian Lague
    public Vector<double> Derivative(Vector<double> input)
    {
        var result = input.PointwiseExp();
        var expSum = result.Sum();
        result.MapInplace(ex => (ex * expSum - ex * ex) / (expSum * expSum));
        return result;
    }

    public Vector<double> DerivativeAlt(Vector<double> input)
    {
        var softmax = Activate(input);
        softmax.MapInplace(softmax => softmax * (1-softmax));
        return softmax;
    }
}
