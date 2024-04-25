using Simple.Training.Cost;

namespace Simple.Training;

public sealed class SimpleTrainingConfig<TInput, TOutput> : TrainingConfig<TInput, TOutput> {
    public required DataPoint<TInput, TOutput>[] TrainingSet { get; init; }
    public required DataPoint<TInput, TOutput>[] TestSet { get; init; }
    public int TrainingBatchSize { get; init; } = -1;
    public int TestBatchSize { get; init; } = -1;

    public override IEnumerable<DataPoint<TInput, TOutput>> GetNextTrainingBatch(){
        var set = TrainingBatchSize < 1 ? TrainingSet : TrainingSet.GetRandomElements(TrainingBatchSize, RandomSource);
        return set.Select(data => new DataPoint<TInput, TOutput>(InputNoise.Apply(data.Input), data.Expected));
    }

    public override IEnumerable<DataPoint<TInput, TOutput>> GetNextTestBatch(){
        var set =  TestBatchSize < 1 ? TestSet : TestSet.GetRandomElements(TestBatchSize, RandomSource);
        return set.Select(data => new DataPoint<TInput, TOutput>(InputNoise.Apply(data.Input), data.Expected));
    }
}

public abstract class TrainingConfig<TInput, TOutput> {
    public required int Iterations { get; init; }
    public int DumpEvaluationAfterIterations { get; init; } = -1;
    public required Number LearnRate { get; init; }
    public Number LearnRateMultiplier { get; init; } = 1;
    public required Number Regularization { get; init; } 
    public required Number Momentum { get; init;}
    //public required int ApplyLearnRateMultiplierAfterIterations { get; init; }
    public IInputDataNoise<TInput> InputNoise { get; init; } = NoInputNoise<TInput>.Instance;
    public required IOutputResolver<TOutput, Number[]> OutputResolver { get; init; }
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public Random RandomSource { get; init; } = Random.Shared;
    public abstract IEnumerable<DataPoint<TInput, TOutput>> GetNextTrainingBatch();
    public abstract IEnumerable<DataPoint<TInput, TOutput>> GetNextTestBatch();
}
