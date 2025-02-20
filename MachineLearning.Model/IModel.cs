using System.Collections.Immutable;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public interface IModel<TArch, TArchSnapshot> : IModel
{
    public IEnumerable<ILayer> Layers { get; }
    public TArch Process(TArch input);
    public TArch Process(TArch input, ImmutableArray<TArchSnapshot> snapshots);
}

public interface IModel
{
    public long ParameterCount { get; }
}
