using System.Collections;

namespace MachineLearning.Data;

public interface ITrainingSet : IEnumerable<Batch>
{
    public int BatchCount { get; }
    public int BatchSize { get; }

    public IEnumerable<Batch> GetBatches();
    public void Reset() { }

    IEnumerator<Batch> IEnumerable<Batch>.GetEnumerator() => GetBatches().GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetBatches().GetEnumerator();
}