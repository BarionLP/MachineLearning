namespace Simple.Network.Activation;
/// <summary>
/// helps with death neurons in <see cref="ReLUActivation"/>
/// </summary>
/// <param name="alpha">slope for x &lt; 0</param>
public sealed class LeakyReLUActivation(Number alpha = 0.01) : IActivation
{
    public static readonly LeakyReLUActivation Instance = new();
    private readonly Number alpha = alpha;
    public Number Function(Number input) => input > 0 ? input : alpha * input;
    public Number Derivative(Number input) => input > 0 ? 1 : alpha;
}
