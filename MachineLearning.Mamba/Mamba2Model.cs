using System.Collections.Immutable;
using MachineLearning.Model;
using MachineLearning.Model.Layer;

namespace MachineLearning.Mamba;

public sealed class Mamba2Model(int layerCount, int contextSize, int dims) : IModel<Vector, Mamba2Layer.Snapshot>
{
    public ImmutableArray<Mamba2Layer> Layers { get; } = [.. Enumerable.Range(0, layerCount).Select(_ => new Mamba2Layer(contextSize, dims))];

    public Vector Process(Vector input)
    {
        return Layers.Aggregate(input, (v, l) => l.Forward(v, (Mamba2Layer.Snapshot) l.CreateSnapshot()));
    }

    public Vector Process(Vector input, ImmutableArray<Mamba2Layer.Snapshot> snapshots)
    {

        return Layers.Zip(snapshots).Aggregate(input, (v, l) => l.First.Forward(v, l.Second));
    }

    public Vector Backward(Vector outputGradient, ImmutableArray<Mamba2Layer.Snapshot> snapshots)
    {

        return Layers.Reverse().Zip(snapshots.Reverse()).Aggregate(outputGradient, (g, l) => l.First.BackwardPass(l.Second, g));
    }

    public long ParameterCount => throw new NotImplementedException();

    IEnumerable<ILayer> IModel<Vector, Mamba2Layer.Snapshot>.Layers => Layers;
}
