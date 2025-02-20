using MachineLearning.Data;
using MachineLearning.Data.Entry;

namespace MachineLearning.Mamba;

public sealed class CumulativeSumTrainingSet(int sequenceLength) : ITrainingSet
{
    private readonly int _sequenceLength = sequenceLength;

    public required int BatchCount { get; init; }
    public required int BatchSize { get; init; }
    public Random Random { get; init; } = Random.Shared;

    public IEnumerable<Batch> GetBatches() => Enumerable.Range(0, BatchCount).Select(_ => Batch.Create(GetTrainingData(), BatchSize));

    public IEnumerable<TrainingData<Vector>> GetTrainingData()
    {
        while (true)
        {
            yield return GenerateSample();
        }
    }

    public TrainingData<Vector> GenerateSample()
    {
        var input = Vector.Create(_sequenceLength);
        for (int t = 0; t < _sequenceLength; t++)
        {
            input[t] = (float)(Random.NextSingle() * 5 + 2.5f);
        }

        var expected = Vector.Create(_sequenceLength);
        var runningSum = 0f;
        for (int t = 0; t < _sequenceLength; t++)
        {
            runningSum += input[t];
            expected[t] = runningSum;
        }

        return new TrainingData<Vector>(input, expected);
    }
}
