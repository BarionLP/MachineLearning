﻿namespace Simple.Network.Activation;

public interface IActivation
{
    public Number Function(Number input);
    public Number Derivative(Number input);

    public Number[] Activate(Number[] input)
    {
        var result = new Number[input.Length];
        foreach (int i in ..input.Length)
        {
            result[i] = Function(input[i]);
        }
        return result;
    }

    public Number[] Derivative(Number[] input)
    {
        var result = new Number[input.Length];
        foreach (int i in ..input.Length)
        {
            result[i] = Derivative(input[i]);
        }
        return result;
    }
}