using System.Runtime.InteropServices;
using MachineLearning.Data;
using MachineLearning.Data.Entry;

namespace MachineLearning.Mamba;

public sealed class CumulativeSumTrainingSet : ITrainingSet
{
    private readonly List<TrainingData<Vector, Vector>> _allData;
    private readonly Random _random;
    private readonly int _sequenceLength;

    public int BatchCount { get; }
    public int BatchSize { get; }

    /// <summary>
    /// Initializes a training set where each sample is:
    ///   Input:  random Vector (length = sequenceLength)
    ///   Target: cumulative sum of the input (same length)
    /// </summary>
    /// <param name="sampleCount">How many sequences to generate in total.</param>
    /// <param name="sequenceLength">Length of each input/target sequence.</param>
    /// <param name="batchSize">Batch size for training.</param>
    /// <param name="minValue">Minimum random input value.</param>
    /// <param name="maxValue">Maximum random input value.</param>
    /// <param name="random">Optional Random instance; if null, uses a shared global random.</param>
    public CumulativeSumTrainingSet(int sampleCount,
                                    int sequenceLength,
                                    int batchSize,
                                    float minValue = -1f,
                                    float maxValue = +1f,
                                    Random? random = null)
    {
        _random = random ?? Random.Shared;
        _sequenceLength = sequenceLength;
        BatchSize = batchSize;

        _allData = new List<TrainingData<Vector, Vector>>(sampleCount);

        for (int i = 0; i < sampleCount; i++)
        {
            // 1) Generate random input x of length = sequenceLength
            var x = Vector.Create(_sequenceLength);
            for (int t = 0; t < _sequenceLength; t++)
            {
                float val = _random.NextSingle() * (maxValue - minValue) + minValue;
                x[t] = val;
            }

            // 2) Compute the cumulative sum y
            var y = Vector.Create(_sequenceLength);
            float runningSum = 0f;
            for (int t = 0; t < _sequenceLength; t++)
            {
                runningSum += x[t];
                y[t] = runningSum;
            }

            _allData.Add(new TrainingData<Vector, Vector>(x, y, y));
        }

        BatchCount = (int)Math.Ceiling((double)_allData.Count / BatchSize);
    }

    /// <summary>
    /// Yield all batches sequentially.
    /// </summary>
    public IEnumerable<Batch> GetBatches()
    {
        // chunk data into consecutive batches
        for (int b = 0; b < BatchCount; b++)
        {
            int startIndex = b * BatchSize;
            yield return Batch.Create(_allData, startIndex, BatchSize);
        }
    }

    public void Reset()
    {
        // If you want to shuffle each epoch, do it here, e.g.:
        _random.Shuffle(CollectionsMarshal.AsSpan(_allData));
    }
}
