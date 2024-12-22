using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Training;

public sealed record TrainingConfig
{
    public required int EpochCount { get; init; }

    public required Optimizer Optimizer { get; init; }
    public bool MultiThread { get; init; } = true;

    public Action<DataSetEvaluation>? EvaluationCallback { get; init; } = null;
    public bool DumpEvaluation => EvaluationCallback is not null;
    public bool DumpEpochEvaluation => DumpEvaluation && !DumpBatchEvaluation;
    public int DumpEvaluationAfterBatches { get; init; } = -1;
    public bool DumpBatchEvaluation => DumpEvaluation && DumpEvaluationAfterBatches > 0;
    public Random RandomSource { get; init; } = Random.Shared;
}