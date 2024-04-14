using System.Text;

namespace Simple;

public sealed class Network {
    internal readonly Layer[] _layers;
    internal Layer _outputLayer => _layers[^1];

    public Network(params int[] layerSizes) {
        _layers = new Layer[layerSizes.Length - 1]; // no need to define the input layer
        foreach(var i in .._layers.Length) {
            _layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
        }
    }

    public Number[] Process(Number[] input) {
        foreach(var layer in _layers) {
            input = layer.Process(input);
        }
        return input;
    }
}

public record DataPoint(Number[] Input, Number[] Expected) : DataPoint<Number[]>(Input, Expected){
    public override string ToString() {
        var sb = new StringBuilder();
        sb.Append('[').AppendCollection(Input, ", ").Append("] => [").AppendCollection(Expected, ", ").Append(']');
        return sb.ToString();
    }
}
public record DataPoint<T>(T Input, T Expected);