using System.IO;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Serialization;

namespace ML.MultiLayerPerceptron;

public sealed class MultiLayerPerceptronModel : IModel<Vector, PerceptronLayer.Snapshot>
{
    public required ImmutableArray<PerceptronLayer> Layers { get; init; }
    public long WeightCount => Layers.Sum(l => l.WeightCount);


    public Vector Process(Vector input)
        => Layers.Aggregate(input, (vector, layer) => layer.Forward(vector, new(layer)));

    public Vector Process(Vector input, ImmutableArray<PerceptronLayer.Snapshot> snapshots)
    {
        Debug.Assert(snapshots.Length == Layers.Length);
        return Layers.Zip(snapshots).Aggregate(input, static (vector, pair) => pair.First.Forward(vector, pair.Second));
    }

    public override string ToString()
        => $"MLP ({Layers.Length} Layers, {WeightCount} Weights)";

    IEnumerable<ILayer> IModel<Vector, PerceptronLayer.Snapshot>.Layers => Layers;

    public static ErrorState Save(MultiLayerPerceptronModel model, BinaryWriter writer)
    {
        writer.Write(model.Layers.Length);
        foreach (var layer in model.Layers)
        {
            var flag = ModelSerializer.SaveLayer(layer, writer);
            if (!OptionsMarshall.IsSuccess(flag))
            {
                return flag;
            }
        }

        return default;
    }

    public static Result<MultiLayerPerceptronModel> Read(BinaryReader reader)
    {
        var layerCount = reader.ReadInt32();
        var layers = new PerceptronLayer[layerCount];
        foreach (var i in ..layerCount)
        {
            var result = ModelSerializer.ReadLayer(reader).Require<PerceptronLayer>();
            if (!result.Branch(out _, out var error))
            {
                return error;
            }
            layers[i] = result.OrThrow();
        }

        return new MultiLayerPerceptronModel
        {
            Layers = [.. layers],
        };
    }
}
