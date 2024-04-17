namespace Simple.Network.Activation;


/// <summary>
/// Produces values between 0 and 1 <br/>
/// Good for Binary Classification <br/>
/// can cause vanishing gradients <br/>
/// high learn rates (1..0.25) <br/>
/// </summary>
public sealed class SigmoidActivation : IActivationMethod {
    public static readonly SigmoidActivation Instance = new();
    public Number Activate(Number input) => 1 / (1 + Math.Exp(-input));
    public Number Derivative(Number input){
        var sigmoid = Activate(input);
        return sigmoid * (1 - sigmoid);
    }
}
