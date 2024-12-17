using MachineLearning.Data;
using MachineLearning.Data.Noise;
using System.Collections.Frozen;

namespace MachineLearning.Samples.MNIST;

public sealed class MNISTDataSet(IEnumerable<DataEntry<double[], int>> data) : ITrainingSet
{
    public bool ShuffleOnReset { get; init; } = true;
    public Random Random { get; init; } = Random.Shared;
    public required int BatchCount { get; init; }
    public IInputDataNoise<double[]> Noise { get; init; } = NoInputNoise<double[]>.Instance;
    public int BatchSize => data.Length / BatchCount;

    private readonly DataEntry<double[], int>[] data = data.ToArray();

    public IEnumerable<Batch> GetBatches()
    {
        var batchSize = BatchSize;
        foreach (var i in ..BatchCount)
        {
            yield return new Batch(batchSize, data.Skip(i * batchSize).Take(batchSize).Select(d =>
            {
                var data = Noise.Apply(d.Input);
                return new TrainingData<double[], int>(data, d.Expected, MNISTModel.Embedder.Embed(data), Expected(d.Expected));
            }));
        }
    }

    public void Reset()
    {
        if (ShuffleOnReset)
        {
            Random.Shuffle(data);
        }
    }

    private readonly FrozenDictionary<int, Vector> _map = new Dictionary<int, Vector>() {
            { 0, Vector.Of([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])},
            { 1, Vector.Of([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])},
            { 2, Vector.Of([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])},
            { 3, Vector.Of([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])},
            { 4, Vector.Of([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])},
            { 5, Vector.Of([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])},
            { 6, Vector.Of([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])},
            { 7, Vector.Of([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])},
            { 8, Vector.Of([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])},
            { 9, Vector.Of([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])},
        }.ToFrozenDictionary();

    private Vector Expected(int output) => _map[output];
}