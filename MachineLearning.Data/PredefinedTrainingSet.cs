using MachineLearning.Data.Entry;

namespace MachineLearning.Data;

public sealed class PredefinedTrainingSet(IEnumerable<TrainingData> data) : ITrainingSet
{
    public bool ShuffleOnReset { get; init; } = true;
    public Random Random { get; init; } = Random.Shared;
    public required int BatchCount { get; init; }
    public int BatchSize => data.Length / BatchCount;

    private readonly TrainingData[] data = data.ToArray();

    public IEnumerable<Batch> GetBatches()
    {
        var batchSize = BatchSize;
        foreach (var i in ..BatchCount)
        {
            yield return Batch.Create(data, i * batchSize, batchSize);
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
