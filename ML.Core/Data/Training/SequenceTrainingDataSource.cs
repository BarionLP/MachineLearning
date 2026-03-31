namespace ML.Core.Data.Training;

public sealed class SequenceTrainingDataSource<T>(IEnumerable<T> sequence) : ITrainingDataSource<T>
{
    public int BatchCount { get; init; } = int.MaxValue;
    public required int BatchSize { get; init; }

    public IEnumerable<IEnumerable<T>> GetBatches()
    {
        using var enumerator = sequence.GetEnumerator();

        foreach (var _ in ..BatchCount)
        {
            if (!enumerator.MoveNext()) yield break;
            yield return YieldBatch(enumerator);
        }

        IEnumerable<T> YieldBatch(IEnumerator<T> enumerator)
        {
            yield return enumerator.Current; // already advanced
            foreach (var _ in ..(BatchSize - 1))
            {
                if (!enumerator.MoveNext()) yield break;
                yield return enumerator.Current;
            }
        }
    }

    public void Reset()
    {

    }
}
