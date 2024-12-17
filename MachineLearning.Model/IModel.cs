using System.Collections.Immutable;

namespace MachineLearning.Model;

public interface IModel<TArch, TArchSnapshot> : IModel
{
    public TArch Process(TArch input);
    public TArch Process(TArch input, ImmutableArray<TArchSnapshot> snapshots);
}

public interface IModel
{
    public long ParameterCount { get; }
}
