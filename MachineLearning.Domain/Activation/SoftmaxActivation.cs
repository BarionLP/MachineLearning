namespace MachineLearning.Domain.Activation;

public sealed class SoftmaxActivation : IActivationMethod
{
    public static readonly SoftmaxActivation Instance = new();

    public void Activate(Vector input, Vector result) {
        input.Map(result, Math.Exp);   
        var sum = result.Sum();
        result.MapInPlace(ex => ex/sum); //TODO: simd
    }

    // adapted from Sebastian Lague
    public void Derivative(Vector input, Vector result) {
        input.Map(result, Math.Exp);
        var expSum = result.Sum();
        result.MapInPlace(ex => (ex * expSum - ex * ex) / (expSum * expSum));
    }

    // ChatGPT (same graph when plotted) TODO: is it faster?
    public void DerivativeAlt(Vector input, Vector result)
    {
        Activate(input, result);
        result.MapInPlace(softmax => softmax * (1-softmax));
    }
}
