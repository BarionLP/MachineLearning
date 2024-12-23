using System.Collections;
using MachineLearning.Data.Entry;

namespace MachineLearning.Data;

public sealed record Batch(IEnumerable<TrainingData> DataPoints) : IEnumerable<TrainingData>
{
    public IEnumerable<TrainingData> DataPoints { get; private set; } = DataPoints;

    public static Batch Create(IEnumerable<TrainingData> source, int startIndex, int batchSize)
    => Create(source.Skip(startIndex), batchSize);

    public static Batch Create(IEnumerable<TrainingData> source, int batchSize)
        => new(source.Take(batchSize));

    public static Batch CreateRandom(ICollection<TrainingData> source, int batchSize, Random? random = null)
        => new(source.GetRandomElements(batchSize, random));

    public IEnumerator<TrainingData> GetEnumerator() => DataPoints.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}