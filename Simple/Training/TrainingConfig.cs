using Simple.Training.Cost;

namespace Simple.Training;

public sealed class TrainingConfig<TInput, TOutput> {
    public required DataPoint<TInput, TOutput>[] TrainingSet { get; init; }
    public required DataPoint<TInput, TOutput>[] TestSet { get; init; }
    public required int Iterations { get; init; }
    public required int DumpEvaluationAfterIterations { get; init; }
    public int BatchSize { get; init; } = -1;
    public required Number LearnRate { get; init; }
    public Number LearnRateMultiplier { get; init; } = 1;
    //public required int ApplyLearnRateMultiplierAfterIterations { get; init; }
    public required IOutputResolver<TOutput, Number[]> OutputResolver { get; init; }
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public Random RandomSource { get; init; } = Random.Shared;
    public IEnumerable<DataPoint<TInput, TOutput>> GetNextTrainingBatch() => BatchSize < 1 ? TrainingSet : TrainingSet.GetRandomElements(BatchSize, RandomSource);
}
