using System.IO;
using System.Runtime.InteropServices;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Serialization;

namespace MachineLearning.Mamba;

public sealed class Mamba2Model(int layerCount, int contextSize, int dims) : IModel<Vector, Mamba2ScalarLayer.Snapshot>
{
    public ImmutableArray<Mamba2ScalarLayer> Layers { get; } = [.. Enumerable.Range(0, layerCount).Select(_ => new Mamba2ScalarLayer(contextSize, dims))];

    public Vector Process(Vector input)
    {
        return Layers.Aggregate(input, (v, l) => l.Forward(v, (Mamba2ScalarLayer.Snapshot)LayerSnapshots.Get(l)));
    }

    public Vector Process(Vector input, ImmutableArray<Mamba2ScalarLayer.Snapshot> snapshots)
    {

        return Layers.Zip(snapshots).Aggregate(input, (v, l) => l.First.Forward(v, l.Second));
    }

    // public Vector Backward(Vector outputGradient, ImmutableArray<Mamba2ScalarLayer.Snapshot> snapshots)
    // {
    //     return Layers.Reverse().Zip(snapshots.Reverse()).Aggregate(outputGradient, (g, l) => l.First.BackwardPass(l.Second, g));
    // }

    public long WeightCount => Layers.Sum(l => l.WeightCount);
    public override string ToString() => $"Mamba 2 (Scalar) ({WeightCount})";

    IEnumerable<ILayer> IModel<Vector, Mamba2ScalarLayer.Snapshot>.Layers => Layers;
}

public sealed class Mamba2VectorModel(EmbeddingLayer inputLayer, ImmutableArray<Mamba2VectorLayer> hiddenLayers, UnEmbeddingLayer outputLayer) : IEmbeddedModel<int[], int>
{
    public EmbeddingLayer InputLayer { get; } = inputLayer;
    public ImmutableArray<Mamba2VectorLayer> HiddenLayers { get; } = hiddenLayers;
    public UnEmbeddingLayer OutputLayer { get; } = outputLayer;

    public int ContextSize => InputLayer.ContextSize;

    public Mamba2VectorModel(int layerCount, int tokenCount, int contextSize, int stateDimensions, int embeddingDimensions)
        : this(new EmbeddingLayer(tokenCount, contextSize, embeddingDimensions), [.. Enumerable.Range(0, layerCount).Select(_ => new Mamba2VectorLayer(contextSize, stateDimensions, embeddingDimensions))], new UnEmbeddingLayer(tokenCount, contextSize, embeddingDimensions)) { }

    public (Matrix, int) Process(int[] input)
    {
        return OutputLayer.Forward(HiddenLayers.Aggregate(InputLayer.Forward(input, (EmbeddingLayer.Snapshot)LayerSnapshots.Get(InputLayer)), (v, l) => l.Forward(v, (Mamba2VectorLayer.Snapshot)LayerSnapshots.Get(l))), (UnEmbeddingLayer.Snapshot)LayerSnapshots.Get(OutputLayer));
    }

    public (Matrix, int) Process(int[] input, ImmutableArray<ILayerSnapshot> snapshots)
    {
        Debug.Assert(snapshots.Length == HiddenLayers.Length + 2);
        return OutputLayer.Forward(HiddenLayers.Zip(snapshots.Skip(1).Take(HiddenLayers.Length).Cast<Mamba2VectorLayer.Snapshot>()).Aggregate(InputLayer.Forward(input, (EmbeddingLayer.Snapshot)snapshots[0]), (v, l) => l.First.Forward(v, l.Second)), (UnEmbeddingLayer.Snapshot)snapshots[^1]);
    }

    public void Initialize(Random? random = null)
    {
        new EmbeddingLayer.Initializer(random).Initialize(InputLayer);
        new UnEmbeddingLayer.Initializer(random).Initialize(OutputLayer);
        var initer = new Mamba2VectorLayer.Initializer(random);
        HiddenLayers.Consume(initer.Initialize);
    }

    // public Vector Backward(Matrix outputGradient, ImmutableArray<EmbeddedMamba2Layer.Snapshot> snapshots)
    // {
    //     return Layers.Reverse().Zip(snapshots.Reverse()).Aggregate(outputGradient, (g, l) => l.First.BackwardPass(l.Second, g));
    // }

    public static ErrorState Save(Mamba2VectorModel model, BinaryWriter writer)
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

    public static Result<Mamba2VectorModel> Read(BinaryReader reader)
    {
        var hiddenLayerCount = reader.ReadInt32();

        var input = ModelSerializer.ReadLayer(reader).Require<EmbeddingLayer>(v => new InvalidCastException("Mamba requires an EmbeddingLayer"));
        if (OptionsMarshall.TryGetError(input, out var error1))
        {
            return error1;
        }

        var hiddenLayers = new Mamba2VectorLayer[hiddenLayerCount];
        foreach (var i in ..hiddenLayerCount)
        {
            var layer = ModelSerializer.ReadLayer(reader).Require<Mamba2VectorLayer>(v => new InvalidCastException("Mamba requires EmbeddedMamba2Layer"));
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

        return new Mamba2VectorModel(input.OrThrow(), ImmutableCollectionsMarshal.AsImmutableArray(hiddenLayers), output.OrThrow());
    }

    public long WeightCount => InputLayer.WeightCount + HiddenLayers.Sum(l => l.WeightCount) + OutputLayer.WeightCount;
    public override string ToString() => $"Mamba 2 (Vector) ({WeightCount})";


    (int prediction, float confidence) IEmbeddedModel<int[], int>.Process(int[] input)
    {
        var (weights, result) = Process(input);
        return (result, weights.RowSpan(weights.RowCount - 1)[result]);
    }

    // IEnumerable<ILayer> IModel<Matrix, EmbeddedMamba2Layer.Snapshot>.Layers => Layers;
}
