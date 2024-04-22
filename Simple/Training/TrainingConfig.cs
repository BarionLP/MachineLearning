using Simple.Training.Cost;

namespace Simple.Training;

public sealed class SimpleTrainingConfig<TInput, TOutput> : TrainingConfig<TInput, TOutput> {
    public required DataPoint<TInput, TOutput>[] TrainingSet { get; init; }
    public required DataPoint<TInput, TOutput>[] TestSet { get; init; }
    public int TrainingBatchSize { get; init; } = -1;
    public int TestBatchSize { get; init; } = -1;

    public override IEnumerable<DataPoint<TInput, TOutput>> GetNextTrainingBatch() => TrainingBatchSize < 1 ? TrainingSet : TrainingSet.GetRandomElements(TrainingBatchSize, RandomSource);
    public override IEnumerable<DataPoint<TInput, TOutput>> GetNextTestBatch() => TestBatchSize < 1 ? TestSet : TestSet.GetRandomElements(TestBatchSize, RandomSource);
}

public abstract class TrainingConfig<TInput, TOutput> {
    public required int Iterations { get; init; }
    public int DumpEvaluationAfterIterations { get; init; } = -1;
    public required Number LearnRate { get; init; }
    public Number LearnRateMultiplier { get; init; } = 1;
    //public required int ApplyLearnRateMultiplierAfterIterations { get; init; }
    public IInputDataNoise<Number> InputNoise { get; init; } = NoInputNoise<Number>.Instance;
    public required IOutputResolver<TOutput, Number[]> OutputResolver { get; init; }
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public Random RandomSource { get; init; } = Random.Shared;
    public abstract IEnumerable<DataPoint<TInput, TOutput>> GetNextTrainingBatch();
    public abstract IEnumerable<DataPoint<TInput, TOutput>> GetNextTestBatch();
}
