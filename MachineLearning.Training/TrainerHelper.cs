using MachineLearning.Data;
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
        //sb.AppendLine($"{TrainingSet.Length} Training Entries");
        //sb.AppendLine($"{TestSet.Length} Test Entries");
        sb.AppendLine($"using {config.Optimizer.GetType().Name}");
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