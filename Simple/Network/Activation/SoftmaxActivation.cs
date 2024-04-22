using Simple.Network.Activation;

namespace Simple;

public sealed class SoftmaxActivation : IActivationMethod<Number>{
    public static readonly SoftmaxActivation Instance = new();

    public Number[] Activate(Number[] input){
        //var maxInput = input.Max(); // Subtracting max for numerical stability (update derivative!!!)
        var result = new Number[input.Length];
        
        foreach(var i in ..input.Length){
            result[i] = Math.Exp(input[i]/* -maxInput */);
        }

        var sum = result.Sum();

        foreach (var i in ..input.Length){
            result[i] = result[i]/sum;
        }

        return result;
    }

    // adapted from Sebastian Lague
    public Number[] Derivative(Number[] input){
        var result = new Number[input.Length];

        foreach (var i in ..input.Length){
            result[i] = Math.Exp(input[i]);
        }
        var expSum = result.Sum();

        foreach(var i in ..result.Length){
            var ex = result[i];
            result[i] = (ex * expSum - ex * ex) / (expSum * expSum);
        }

        return result;
    }
}

public sealed class ChatGPTSoftmaxActivation : IActivationMethod<Number>{
    public static readonly ChatGPTSoftmaxActivation Instance = new();
    public Number[] Activate(Number[] input){
        var maxInput = input.Max();
        var result = new Number[input.Length];
        var sum = 0.0;

        for (var i = 0; i < input.Length; i++){
            result[i] = Math.Exp(input[i] - maxInput);
            sum += result[i];
        }

        for (var i = 0; i < input.Length; i++){
            result[i] /= sum;
        }

        return result;
    }

    public Number[] Derivative(Number[] input){
        var softmax = Activate(input);
        var result = new Number[input.Length];

        for (var i = 0; i < input.Length; i++){
            for (var j = 0; j < input.Length; j++){
                if (i == j){
                    result[i] += softmax[i] * (1 - softmax[j]);
                } else{
                    result[i] -= softmax[i] * softmax[j];
                }
            }
        }

        return result;
    }
}
