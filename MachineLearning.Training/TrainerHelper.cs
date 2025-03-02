using MachineLearning.Data;
using MachineLearning.Training.Evaluation;
using System.Text;

namespace MachineLearning.Training;

public static class TrainerHelper
{
    public static void TrainConsole<TModel>(this ITrainer<TModel> trainer, bool cancelable = true)
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

        Console.WriteLine($"Training {trainer.Model}");
        Console.WriteLine(GenerateTrainingOverview(trainer.Config, trainer.TrainingSet));
        Console.WriteLine("Starting Training...");
        trainer.Train(cts.Token);
        cts.Cancel();
        Console.WriteLine("Training Done!");
    }

    public static void Train<TModel>(this ITrainer<TModel> trainer, CancellationToken? token = null)
    {
        trainer.Config.Optimizer.Init();
        trainer.FullReset();
        var cachedEvaluation = DataSetEvaluationResult.ZERO;
        foreach (var (epochIndex, epoch) in GetEpochs(trainer.TrainingSet, trainer.Config.EpochCount).Index())
        {
            foreach (var (batchIndex, batch) in epoch.Index())
            {
                cachedEvaluation += trainer.TrainAndEvaluate(batch);
                if (trainer.Config.DumpBatchEvaluation && batchIndex % trainer.Config.DumpEvaluationAfterBatches == 0 || batchIndex + 1 == epoch.BatchCount && trainer.Config.DumpEpochEvaluation)
                {
                    trainer.Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = cachedEvaluation });
                    cachedEvaluation = DataSetEvaluationResult.ZERO;
                }
                trainer.Config.Optimizer.OnBatchCompleted();

                if (token?.IsCancellationRequested is true)
                {
                    trainer.Config.Optimizer.OnEpochCompleted();
                    return;
                }

                TrainingEvaluationContext GetContext() => new()
                {
                    CurrentBatch = batchIndex + 1,
                    MaxBatch = epoch.BatchCount,
                    CurrentEpoch = epochIndex + 1,
                    MaxEpoch = trainer.Config.EpochCount,
                    LearnRate = trainer.Config.Optimizer.LearningRate,
                };
            }

            trainer.Config.Optimizer.OnEpochCompleted();
        }
    }

    public static IEnumerable<Epoch> GetEpochs(ITrainingSet trainingSet, int epochCount)
    {
        foreach (var _ in ..epochCount)
        {
            trainingSet.Reset();
            yield return new Epoch(trainingSet.BatchCount, trainingSet.GetBatches());
        }
    }

    public static string GenerateTrainingOverview(TrainingConfig config, ITrainingSet trainingSet)
    {
        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine("Training Info:");
        sb.AppendLine($"using {config.Optimizer.GetType().Name} ({config.Threading})");
        sb.AppendLine("Training for");
        sb.AppendLine($" - {config.EpochCount} epochs");
        sb.AppendLine($"  - {trainingSet.BatchCount} batches");
        sb.AppendLine($"   - {trainingSet.BatchSize} entries");

        if (config.DumpEvaluation)
        {
            if (config.DumpBatchEvaluation)
            {
                if (config.DumpEvaluationAfterBatches == 1)
                {
                    sb.AppendLine("Dumping every batch");
                }
                else
                {
                    sb.AppendLine($"Dumping every {config.DumpEvaluationAfterBatches} batches");
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