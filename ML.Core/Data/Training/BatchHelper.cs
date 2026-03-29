namespace ML.Core.Data.Training;

public static class BatchHelper
{
    public static IEnumerable<T> Create<T>(IEnumerable<T> source, int startIndex, int batchSize)
        => Create(source.Skip(startIndex), batchSize);

    public static IEnumerable<T> Create<T>(IEnumerable<T> source, int batchSize)
        => source.Take(batchSize);

    public static IEnumerable<T> CreateRandom<T>(ICollection<T> source, int batchSize, Random? random = null)
        => source.GetRandomElements(batchSize, random);
}