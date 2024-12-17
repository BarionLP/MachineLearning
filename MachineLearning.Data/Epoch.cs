using System.Collections;

namespace MachineLearning.Data;

public sealed record Epoch(int BatchCount, IEnumerable<Batch> Batches) : IEnumerable<Batch>
{
    public IEnumerator<Batch> GetEnumerator() => Batches.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
