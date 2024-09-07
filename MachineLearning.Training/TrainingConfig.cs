using MachineLearning.Data.Entry;
using MachineLearning.Data.Noise;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Training;

public sealed record TrainingConfig<TInput, TOutput>
{
    public required DataEntry<TInput, TOutput>[] TrainingSet { get; init; }
    public required DataEntry<TInput, TOutput>[] TestSet { get; init; }
    public bool ShuffleTrainingSetPerEpoch { get; init; } = false;

    public required int EpochCount { get; init; }
    public required int BatchCount { get; init; }
    public int BatchSize => TrainingSet.Length / BatchCount;

    public required IOptimizer Optimizer { get; init; }

    public IInputDataNoise<TInput> InputNoise { get; init; } = NoInputNoise<TInput>.Instance;
    public required IOutputResolver<TOutput> OutputResolver { get; init; }

    public Action<DataSetEvaluation>? EvaluationCallback { get; init; } = null;
    public bool DumpEvaluation => EvaluationCallback is not null;
    public bool DumpEpochEvaluation => DumpEvaluation && !DumpBatchEvaluation;
    public int DumpEvaluationAfterBatches { get; init; } = -1;
    public bool DumpBatchEvaluation => DumpEvaluation && DumpEvaluationAfterBatches > 0;
    public Random RandomSource { get; init; } = Random.Shared;

    public Epoch<TInput, TOutput> GetEpoch()
    {
        if(ShuffleTrainingSetPerEpoch)
        {
            Shuffle(TrainingSet);
        }

        return new Epoch<TInput, TOutput>(BatchCount, GetBatches());

        IEnumerable<Batch<TInput, TOutput>> GetBatches()
        {
            foreach(var i in ..BatchCount)
            {
                yield return GetTrainingBatch(i * BatchSize, BatchSize).ApplyNoise(InputNoise);
            }
        }

        void Shuffle<T>(T[] array)
        {
            int n = array.Length;
            for(int i = n - 1; i > 0; i--)
            {
                int j = RandomSource.Next(i + 1);
                (array[i], array[j]) = (array[j], array[i]);
            }
        }
    }

    public Batch<TInput, TOutput> GetRandomTrainingBatch() => GetRandomTrainingBatch(BatchSize);
    public Batch<TInput, TOutput> GetRandomTrainingBatch(int batchSize)
        => Batch.CreateRandom(TrainingSet, batchSize, RandomSource);
    public Batch<TInput, TOutput> GetTrainingBatch(int startIndex, int batchSize)
        => Batch.Create(TrainingSet, startIndex, batchSize);

    public Batch<TInput, TOutput> GetRandomTestBatch() => GetRandomTestBatch(BatchSize);
    public Batch<TInput, TOutput> GetRandomTestBatch(int batchSize)
        => Batch.CreateRandom(TestSet, batchSize, RandomSource);
}
