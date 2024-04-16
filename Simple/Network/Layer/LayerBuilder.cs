using Simple.Network.Activation;

namespace Simple.Network.Layer;

public sealed class LayerBuilder(int inputNodeCount, int outputNodeCount) {
    public Number[,] Weights { get; } = new Number[inputNodeCount, outputNodeCount];
    public Number[] Biases { get; } = new Number[outputNodeCount];
    public IActivationMethod ActivationMethod { get; set; } = SigmoidActivation.Instance;
    public bool IsRecorded { get; set; } = false;
    
    public LayerBuilder SetActivationMethod(IActivationMethod activationMethod) { 
        ActivationMethod = activationMethod;
        return this;
    }
    public LayerBuilder Initialize(int defaultValue) {
        foreach(int outputNodeIndex in ..Biases.Length) {
            foreach(int inputNodeIndex in ..Weights.GetLength(0)) {
                Weights[inputNodeIndex, outputNodeIndex] = defaultValue;
            }
            Biases[outputNodeIndex] = defaultValue;
        }
        return this;
    }
    public LayerBuilder InitializeRandom(Random? random = null) {
        var sqrtInputNodeCount = Math.Sqrt(Weights.GetLength(0));
        random ??= Random.Shared;
        foreach(int outputNodeIndex in ..Biases.Length) {
            foreach(int inputNodeIndex in ..Weights.GetLength(0)) {
                Weights[inputNodeIndex, outputNodeIndex] = RandomInNormalDistribution(random, 0, 1) / sqrtInputNodeCount;
            }
            Biases[outputNodeIndex] = RandomInNormalDistribution(random, 0, 0.1);
        }
        return this;

        static Number RandomInNormalDistribution(Random random, Number mean, Number standardDeviation) {
            var x1 = 1 - random.NextDouble();
            var x2 = 1 - random.NextDouble();

            var y1 = Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
            return y1 * standardDeviation + mean;
        }
    }

    public LayerBuilder Record() => SetRecorded(true);
    public LayerBuilder SetRecorded(bool isRecorded) {
        IsRecorded = isRecorded;
        return this;
    }

    public ILayer<Number> Build() => IsRecorded switch {
        false => new SimpleLayer(Weights, Biases, ActivationMethod),
        true => new RecordingLayer(Weights, Biases, ActivationMethod),
    };

    public static LayerBuilder Of(ILayer<Number> layer) {
        var builder = new LayerBuilder(layer.InputNodeCount, layer.OutputNodeCount)
            .SetActivationMethod(layer.ActivationMethod);
        
        if(layer is RecordingLayer) builder.Record();
        
        return builder;
    }
}
