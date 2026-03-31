namespace ML.Core.Data.Training;

public sealed class MemoryTrainingDataSource<T>(IEnumerable<T> data) : ITrainingDataSource<T>
{
    public bool ShuffleOnReset { get; init; } = true;
    public Random Random { get; init; } = Random.Shared;
    public required int BatchCount { get; init; }
    public int BatchSize => data.Length / BatchCount;

    private readonly T[] data = [.. data];

    public IEnumerable<IEnumerable<T>> GetBatches()
    {
        var batchSize = BatchSize;
        foreach (var i in ..BatchCount)
        {
            yield return BatchHelper.Create(data, i * batchSize, batchSize);
        }
    }

    public void Reset()
    {
        if (ShuffleOnReset)
        {
            Random.Shuffle(data);
        }
    }
}