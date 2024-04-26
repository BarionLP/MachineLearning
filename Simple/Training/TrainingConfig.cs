﻿using System.Collections;
using Simple.Training.Cost;
using Simple.Training.Evaluation;

namespace Simple.Training;

public sealed class TrainingConfig<TInput, TOutput> {
    public required DataPoint<TInput, TOutput>[] TrainingSet { get; init; }
    public required DataPoint<TInput, TOutput>[] TestSet { get; init; }

    public required int EpochCount  { get; init; }
    public required int BatchSize { get; init; }
    public required Number LearnRate { get; init; }
    public Number LearnRateMultiplier { get; init; } = 1;
    public required Number Regularization { get; init; } 
    public required Number Momentum { get; init; }

    public IInputDataNoise<TInput> InputNoise { get; init; } = NoInputNoise<TInput>.Instance;
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;
    public required IOutputResolver<TOutput, Number[]> OutputResolver { get; init; }

    public Action<NetworkEvaluation>? EvaluationCallback { get; init; } = null;
    public bool DumpEvaluation => EvaluationCallback is not null;
    public bool DumpEpochEvaluation => DumpEvaluation && !DumpBatchEvaluation;
    public int DumpEvaluationAfterBatches { get; init; } = -1;
    public bool DumpBatchEvaluation => DumpEvaluation && DumpEvaluationAfterBatches > 0;
    public Random RandomSource { get; init; } = Random.Shared;

    public Epoch<TInput, TOutput> GetEpoch(){
        return new Epoch<TInput, TOutput>((int)MathF.Ceiling(TrainingSet.Length/(float) BatchSize), GetBatches());

        IEnumerable<Batch<TInput, TOutput>> GetBatches(){
            var index = 0;
            while(index < TrainingSet.Length){
                var batchSize = Math.Min(TrainingSet.Length-index, BatchSize);
                yield return GetTrainingBatch(index, batchSize).ApplyNoise(InputNoise);
                index += batchSize;
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
