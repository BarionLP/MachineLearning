using System.Collections.Immutable;
using MachineLearning.Model;

namespace MachineLearning.Mamba;

public sealed class Mamba2Model(int layerCount, int contextSize, int dims) : IModel<Vector, Mamba2Layer.Snapshot>
{
    public ImmutableArray<Mamba2Layer> Layers { get; } = [.. Enumerable.Range(0, layerCount).Select(_ => new Mamba2Layer(contextSize, dims))];

    public Vector Process(Vector input)
    {

        throw new NotImplementedException();
    }

    public Vector Process(Vector input, ImmutableArray<Mamba2Layer.Snapshot> snapshots)
    {

        return Layers.Zip(snapshots).Aggregate(input, (v, l) => l.First.Forward(v, l.Second));
    }

    public long ParameterCount => throw new NotImplementedException();
}
