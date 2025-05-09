﻿namespace MachineLearning.Model.Activation;

/// <summary>
/// outputs values between -1 and 1 (adjust output resolver!) <br/>
/// can cause vanishing gradients (often better than <see cref="SigmoidActivation"/> because centered around zero)
/// </summary>
public sealed class TanhActivation : ISimpleActivationMethod
{
    public static readonly TanhActivation Instance = new();
    public Weight Activate(Weight input) => float.Tanh(input);
    public Weight Derivative(Weight input) => 1 - float.Pow(float.Tanh(input), 2);
}
