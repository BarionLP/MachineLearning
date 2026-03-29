using ML.Core.Evaluation;

namespace ML.Core.Training;

public sealed record TrainingConfig
{
    public required int EpochCount { get; init; }

    public required Optimizer Optimizer { get; init; }
    public ThreadingMode Threading { get; init; } = ThreadingMode.Full;

    public Action<TrainingEvaluationResult>? EvaluationCallback { get; init; } = null;
    public bool EvaluationCallbackEnabled => EvaluationCallback is not null;
    public bool EpochEvaluationEnabled => EvaluationCallbackEnabled && !BatchEvaluationEnabled;
    public int EvaluationCallbackAfterBatches { get; init; } = -1;
    public bool BatchEvaluationEnabled => EvaluationCallbackEnabled && EvaluationCallbackAfterBatches > 0;
}
