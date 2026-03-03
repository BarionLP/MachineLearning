using ML.Core.Evaluation;
using ML.Core.Modules;
using ML.Core.Training.Data;

namespace ML.Core.Training;

public sealed class EmbeddedModuleTrainer<TIn, TArch, TOut>
    where TArch : ITensorLike<TArch>
{
    public EmbeddedModule<TIn, TArch, TOut> Module { get; }
    public TrainingConfig Config { get; }
    public required ITrainingDataSource<TIn, TArch, TOut> TrainingData { get; init; }
    public required ICostFunction<TArch> CostFunction { get; init; }

    private Optimizer Optimizer => Config.Optimizer;
    private readonly IModuleOptimizer moduleOptimizer;

    public EmbeddedModuleTrainer(EmbeddedModule<TIn, TArch, TOut> module, TrainingConfig config)
    {
        Module = module;
        Config = config;
        moduleOptimizer = Optimizer.CreateModuleOptimizer(module);
    }

    public EvaluationResult RunBatch(IEnumerable<TrainingEntry<TIn, TArch, TOut>> batch)
    {
        var timeStamp = Stopwatch.GetTimestamp();

        var gradients = Module.CreateGradients();

        int correctCount = 0;
        int totalCount = 0;
        Weight totalCost = 0;

        foreach (var entry in batch)
        {
            var (output, condfidence, cost) = RunEntry(entry, gradients);
            if (EqualityComparer<TOut>.Default.Equals(output, entry.ExpectedValue))
            {
                correctCount++;
            }

            // TODO: track confidence
            totalCount++;
            totalCost += cost;
        }

        moduleOptimizer.Apply(gradients);

        return new()
        {
            TotalCount = totalCount,
            CorrectCount = correctCount,
            TotalCost = totalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };
    }

    private (TOut output, Weight confidence, Weight cost) RunEntry(TrainingEntry<TIn, TArch, TOut> entry, EmbeddedModule<TIn, TArch, TOut>.Gradients gradients)
    {
        var snapshot = Module.CreateSnapshot(); // TODO: reuse snapshot

        var (output, condfidence, outputWeights) = Module.Forward(entry.InputValue, snapshot);

        var outputGradient = CostFunction.Derivative(outputWeights, entry.ExpectedWeights); // TODO: reuse Tensor

        var inputGradient = Module.Backward(outputGradient, snapshot, gradients);

        return (output, condfidence, CostFunction.TotalCost(outputWeights, entry.ExpectedWeights));
    }
}