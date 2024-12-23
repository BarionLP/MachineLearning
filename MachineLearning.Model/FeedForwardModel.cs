using System.Collections.Immutable;
using System.Diagnostics;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model;

public sealed class FeedForwardModel : IModel<Vector, LayerSnapshots.Simple>
{
    public required ImmutableArray<FeedForwardLayer> Layers { get; init; }
    public long ParameterCount => Layers.Sum(l => l.ParameterCount);


    public Vector Process(Vector input)
        => Layers.Aggregate(input, (vector, layer) => layer.Forward(vector));

    public Vector Process(Vector input, ImmutableArray<LayerSnapshots.Simple> snapshots)
    {
        Debug.Assert(snapshots.Length == Layers.Length);
        return Layers.Zip(snapshots).Aggregate(input, static (vector, pair) => pair.First.Forward(vector, pair.Second));
    }

    public override string ToString()
        => $"Feed Forward Model ({Layers.Length} Layers, {ParameterCount} Weights)";
}
