namespace MachineLearning.Domain.Activation;


/// <summary>
/// Produces values between 0 and 1 <br/>
/// Good for Binary Classification <br/>
/// can cause vanishing gradients <br/>
/// high learn rates (1..0.25) <br/>
/// </summary>
public sealed class SigmoidActivation : ISimpleActivationMethod
{
    public static readonly SigmoidActivation Instance = new();
    public Weight Activate(Weight input) => 1 / (1 + Math.Exp(-input));
    public Weight Derivative(Weight input)
    {
        var sigmoid = Activate(input);
        return sigmoid * (1 - sigmoid);
    }
}
