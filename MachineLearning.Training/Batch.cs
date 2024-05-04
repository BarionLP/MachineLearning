using System.Collections;
using MachineLearning.Data.Entry;
using MachineLearning.Data.Noise;

namespace MachineLearning.Training;

public record Batch<TInput, TOutput>(int Size, IEnumerable<DataEntry<TInput, TOutput>> DataPoints) : IEnumerable<DataEntry<TInput, TOutput>>
{
    public IEnumerable<DataEntry<TInput, TOutput>> DataPoints { get; private set; } = DataPoints;
    public Batch<TInput, TOutput> ApplyNoise(IInputDataNoise<TInput> inputNoise)
    {
        DataPoints = DataPoints.Select(data => new DataEntry<TInput, TOutput>(inputNoise.Apply(data.Input), data.Expected));
        return this;
    }

    public IEnumerator<DataEntry<TInput, TOutput>> GetEnumerator() => DataPoints.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}


public static class Batch
{
    public static Batch<TInput, TOutput> Create<TInput, TOutput>(IEnumerable<DataEntry<TInput, TOutput>> source, int startIndex, int batchSize)
        => Create(source.Skip(startIndex), batchSize);

    public static Batch<TInput, TOutput> Create<TInput, TOutput>(IEnumerable<DataEntry<TInput, TOutput>> source, int batchSize)
        => new(batchSize, source.Take(batchSize));

    public static Batch<TInput, TOutput> CreateRandom<TInput, TOutput>(ICollection<DataEntry<TInput, TOutput>> source, int batchSize, Random? random = null)
        => new(batchSize, source.GetRandomElements(batchSize, random));

}