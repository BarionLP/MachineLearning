using System.Text;
using System.Threading;
using ML.Core.Evaluation;
using ML.Core.Modules;
using ML.Core.Training.Data;

namespace ML.Core.Training;

public sealed class EmbeddedModuleTrainer<TIn, TArch, TOut>
    where TArch : ITensorLike<TArch>
{
    public EmbeddedModule<TIn, TArch, TOut> Module { get; }
    public TrainingConfig Config { get; }
    public required ITrainingDataSource<TrainingEntry<TIn, TArch, TOut>> TrainingData { get; init; }
    public required ICostFunction<TArch> CostFunction { get; init; }

    private Optimizer Optimizer => Config.Optimizer;
    private readonly IModuleOptimizer moduleOptimizer;

    public EmbeddedModuleTrainer(EmbeddedModule<TIn, TArch, TOut> module, TrainingConfig config)
    {
        Module = module;
        Config = config;
        moduleOptimizer = Optimizer.CreateModuleOptimizer(module);
    }

    public void TrainConsole(bool cancelable = true)
    {
        using var cts = new CancellationTokenSource();
        if (cancelable)
        {
            Task.Run(async () =>
            {
                while (!cts.IsCancellationRequested)
                {
                    if (Console.KeyAvailable && Console.ReadKey(intercept: true).Key == ConsoleKey.C)
                    {
                        Console.WriteLine("Canceling...");
                        cts.Cancel();
                        break;
                    }
                    await Task.Delay(500);
                }
            });
        }

        Console.WriteLine($"Training {Module}");
        Console.WriteLine(GenerateTrainingOverview(Config, TrainingData.BatchCount, TrainingData.BatchSize));
        Console.WriteLine("Starting Training...");
        Train(cts.Token);
        cts.Cancel();
        Console.WriteLine("Training Done!");
    }


    public void Train(CancellationToken token = default)
    {
        Optimizer.Init();
        var runningEvaluation = EvaluationResult.ZERO;

        foreach (var epochIndex in ..Config.EpochCount)
        {
            TrainingData.Reset();

            foreach (var (batchIndex, batch) in TrainingData.GetBatches().Index())
            {
                runningEvaluation += RunBatch(batch);

                if ((Config.BatchEvaluationEnabled && batchIndex % Config.EvaluationCallbackAfterBatches is 0)
                    || (batchIndex + 1 == TrainingData.BatchCount && Config.EpochEvaluationEnabled))
                {
                    Config.EvaluationCallback!.Invoke(new TrainingEvaluationResult { Context = GetContext(), Result = runningEvaluation });
                    runningEvaluation = EvaluationResult.ZERO;
                }

                Optimizer.OnBatchCompleted();

                if (token.IsCancellationRequested)
                {
                    Optimizer.OnEpochCompleted();
                    return;
                }

                TrainingEvaluationContext GetContext() => new()
                {
                    CurrentBatch = batchIndex + 1,
                    MaxBatch = TrainingData.BatchCount,
                    CurrentEpoch = epochIndex + 1,
                    MaxEpoch = Config.EpochCount,
                    LearningRate = Optimizer.LearningRate,
                };
            }

            Optimizer.OnEpochCompleted();
        }
    }

    public EvaluationResult RunBatch(IEnumerable<TrainingEntry<TIn, TArch, TOut>> batch)
    {
        var timeStamp = Stopwatch.GetTimestamp();

        var gradients = Module.CreateGradients(); // TODO: reuse

        int correctCount = 0;
        int totalCount = 0;
        Weight totalCost = 0;

        // TODO: threading
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

    public void FullReset()
    {
        moduleOptimizer.FullReset();
    }

    public static string GenerateTrainingOverview(TrainingConfig config, int batchCount, int batchSize)
    {
        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine("Training Info:");
        sb.AppendLine($"using {config.Optimizer.GetType().Name} ({config.Threading})");
        sb.AppendLine("Training for");
        sb.AppendLine($" - {config.EpochCount} epochs");
        sb.AppendLine($"  - {batchCount} batches");
        sb.AppendLine($"   - {batchSize} entries");

        if (config.EvaluationCallbackEnabled)
        {
            if (config.BatchEvaluationEnabled)
            {
                if (config.EvaluationCallbackAfterBatches == 1)
                {
                    sb.AppendLine("Dumping every batch");
                }
                else
                {
                    sb.AppendLine($"Dumping every {config.EvaluationCallbackAfterBatches} batches");
                }
            }
            else
            {
                sb.AppendLine($"Dumping every epoch");
            }
        }

        sb.AppendLine();
        return sb.ToString();
    }
}