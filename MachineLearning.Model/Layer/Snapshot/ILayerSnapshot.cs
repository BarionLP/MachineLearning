namespace MachineLearning.Model.Layer.Snapshot;

public interface ILayerSnapshot
{
    public static ILayerSnapshot Empty { get; } = new EmptySnapshot();
}

file sealed class EmptySnapshot : ILayerSnapshot;
