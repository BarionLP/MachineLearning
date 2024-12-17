using System.Collections;
using MachineLearning.Data.Entry;

namespace MachineLearning.Data;

public record Batch(int Size, IEnumerable<TrainingData> DataPoints) : IEnumerable<TrainingData>
{
    public IEnumerable<TrainingData> DataPoints { get; private set; } = DataPoints;

    public static Batch Create(IEnumerable<TrainingData> source, int startIndex, int batchSize)
    => Create(source.Skip(startIndex), batchSize);

    public static Batch Create(IEnumerable<TrainingData> source, int batchSize)
        => new(batchSize, source.Take(batchSize));

    public static Batch CreateRandom(ICollection<TrainingData> source, int batchSize, Random? random = null)
        => new(batchSize, source.GetRandomElements(batchSize, random));

    public IEnumerator<TrainingData> GetEnumerator() => DataPoints.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}