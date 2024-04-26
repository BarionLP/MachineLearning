using System.Collections;

namespace Simple.Training;

public record Batch<TInput, TOutput>(int Size, IEnumerable<DataPoint<TInput, TOutput>> DataPoints) : IEnumerable<DataPoint<TInput, TOutput>>{
    public IEnumerable<DataPoint<TInput, TOutput>> DataPoints { get; private set; } = DataPoints;
    public Batch<TInput, TOutput> ApplyNoise(IInputDataNoise<TInput> inputNoise){
        DataPoints = DataPoints.Select(data => new DataPoint<TInput, TOutput>(inputNoise.Apply(data.Input), data.Expected));
        return this;
    }

    public IEnumerator<DataPoint<TInput, TOutput>> GetEnumerator() => DataPoints.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}


public static class Batch {
    public static Batch<TInput, TOutput> Create<TInput, TOutput>(IEnumerable<DataPoint<TInput, TOutput>> source, int startIndex, int batchSize)
        => Create(source.Skip(startIndex), batchSize);

    public static Batch<TInput, TOutput> Create<TInput, TOutput>(IEnumerable<DataPoint<TInput, TOutput>> source, int batchSize)
        => new(batchSize, source.Take(batchSize));

    public static Batch<TInput, TOutput> CreateRandom<TInput, TOutput>(ICollection<DataPoint<TInput, TOutput>> source, int batchSize, Random? random = null)
        => new(batchSize, source.GetRandomElements(batchSize, random));

}