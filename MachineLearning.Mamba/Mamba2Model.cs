using System.Collections.Immutable;
using System.Diagnostics;
using System.Runtime.InteropServices;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Serialization;

namespace MachineLearning.Mamba;

public sealed class Mamba2Model(int layerCount, int contextSize, int dims) : IModel<Vector, Mamba2Layer.Snapshot>
{
    public ImmutableArray<Mamba2Layer> Layers { get; } = [.. Enumerable.Range(0, layerCount).Select(_ => new Mamba2Layer(contextSize, dims))];

    public Vector Process(Vector input)
    {
        return Layers.Aggregate(input, (v, l) => l.Forward(v, (Mamba2Layer.Snapshot)LayerSnapshots.Get(l)));
    }

    public Vector Process(Vector input, ImmutableArray<Mamba2Layer.Snapshot> snapshots)
    {

        return Layers.Zip(snapshots).Aggregate(input, (v, l) => l.First.Forward(v, l.Second));
    }

    public Vector Backward(Vector outputGradient, ImmutableArray<Mamba2Layer.Snapshot> snapshots)
    {

        return Layers.Reverse().Zip(snapshots.Reverse()).Aggregate(outputGradient, (g, l) => l.First.BackwardPass(l.Second, g));
    }

    public long ParameterCount => Layers.Sum(l => l.ParameterCount);
    public override string ToString() => $"Mamba 2 (Scalar) ({ParameterCount})";

    IEnumerable<ILayer> IModel<Vector, Mamba2Layer.Snapshot>.Layers => Layers;
}

public sealed class EmbeddedMamba2Model(EmbeddingLayer inputLayer, ImmutableArray<EmbeddedMamba2Layer> hiddenLayers, UnEmbeddingLayer outputLayer) : IEmbeddedModel<int[], int>
{
    public EmbeddingLayer InputLayer { get; } = inputLayer;
    public ImmutableArray<EmbeddedMamba2Layer> HiddenLayers { get; } = hiddenLayers;
    public UnEmbeddingLayer OutputLayer { get; } = outputLayer;

    public EmbeddedMamba2Model(int layerCount, int tokenCount, int contextSize, int stateDimensions, int embeddingDimensions)
        : this(new EmbeddingLayer(tokenCount, contextSize, embeddingDimensions), [.. Enumerable.Range(0, layerCount).Select(_ => new EmbeddedMamba2Layer(contextSize, stateDimensions, embeddingDimensions))], new UnEmbeddingLayer(tokenCount, contextSize, embeddingDimensions)) { }

    public (Vector, int) Process(int[] input)
    {
        return OutputLayer.Forward(HiddenLayers.Aggregate(InputLayer.Forward(input, (EmbeddingLayer.Snapshot)LayerSnapshots.Get(InputLayer)), (v, l) => l.Forward(v, (EmbeddedMamba2Layer.Snapshot)LayerSnapshots.Get(l))), (UnEmbeddingLayer.Snapshot)LayerSnapshots.Get(OutputLayer));
    }

    public (Vector, int) Process(int[] input, ImmutableArray<ILayerSnapshot> snapshots)
    {
        Debug.Assert(snapshots.Length == HiddenLayers.Length + 2);
        return OutputLayer.Forward(HiddenLayers.Zip(snapshots.Skip(1).Take(HiddenLayers.Length).Cast<EmbeddedMamba2Layer.Snapshot>()).Aggregate(InputLayer.Forward(input, (EmbeddingLayer.Snapshot)snapshots[0]), (v, l) => l.First.Forward(v, l.Second)), (UnEmbeddingLayer.Snapshot)snapshots[^1]);
    }

    // public Vector Backward(Matrix outputGradient, ImmutableArray<EmbeddedMamba2Layer.Snapshot> snapshots)
    // {
    //     return Layers.Reverse().Zip(snapshots.Reverse()).Aggregate(outputGradient, (g, l) => l.First.BackwardPass(l.Second, g));
    // }

    public static ErrorState Save(EmbeddedMamba2Model model, BinaryWriter writer)
    {
        writer.Write(model.HiddenLayers.Length);

        if (OptionsMarshall.TryGetError(ModelSerializer.SaveLayer(model.InputLayer, writer), out var error1))
        {
            return error1;
        }

        foreach (var layer in model.HiddenLayers)
        {
            if (OptionsMarshall.TryGetError(ModelSerializer.SaveLayer(layer, writer), out var error2))
            {
                return error2;
            }
        }

        if (OptionsMarshall.TryGetError(ModelSerializer.SaveLayer(model.OutputLayer, writer), out var error3))
        {
            return error3;
        }

        return default;
    }

    public static Result<EmbeddedMamba2Model> Read(BinaryReader reader)
    {
        var hiddenLayerCount = reader.ReadInt32();

        var input = ModelSerializer.ReadLayer(reader).Require<EmbeddingLayer>(v => new InvalidCastException("Mamba requires an EmbeddingLayer"));
        if (OptionsMarshall.TryGetError(input, out var error1))
        {
            return error1;
        }
        var hiddenLayers = new EmbeddedMamba2Layer[hiddenLayerCount];
        foreach (var i in ..hiddenLayerCount)
        {
            var layer = ModelSerializer.ReadLayer(reader).Require<EmbeddedMamba2Layer>(v => new InvalidCastException("Mamba requires EmbeddedMamba2Layer"));
            if (OptionsMarshall.TryGetError(layer, out var error2))
            {
                return error2;
            }
            hiddenLayers[i] = layer.OrThrow();
        }

        var output = ModelSerializer.ReadLayer(reader).Require<UnEmbeddingLayer>(v => new InvalidCastException("Mamba requires an UnEmbeddingLayer"));
        if (OptionsMarshall.TryGetError(output, out var error3))
        {
            return error3;
        }

        return new EmbeddedMamba2Model(input.OrThrow(), ImmutableCollectionsMarshal.AsImmutableArray(hiddenLayers), output.OrThrow());
    }

    public long ParameterCount => InputLayer.ParameterCount + HiddenLayers.Sum(l => l.ParameterCount) + OutputLayer.ParameterCount;
    public override string ToString() => $"Mamba 2 (Vector) ({ParameterCount})";


    (int prediction, float confidence) IEmbeddedModel<int[], int>.Process(int[] input)
    {
        var (weights, result) = Process(input);
        return (result, weights[result]);
    }

    // IEnumerable<ILayer> IModel<Matrix, EmbeddedMamba2Layer.Snapshot>.Layers => Layers;
}
