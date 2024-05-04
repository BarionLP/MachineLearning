using MachineLearning.Model;
using MachineLearning.Training.Evaluation;

namespace MachineLearning.Training;

public sealed class NetworkTrainer<TInput, TOutput>(TrainingConfig<TInput, TOutput> config, RecordingNetwork<TInput, TOutput> network) where TInput : notnull where TOutput : notnull
{
    public TrainingConfig<TInput, TOutput> Config { get; } = config;
    public RecordingNetwork<TInput, TOutput> Network { get; } = network;
    internal NetworkTrainingContext<TInput, TOutput> Context { get; } = new(network, config);

    public ModelTrainingResult Train()
    {
        // for each epoch 
        // train on all batches
        // decay learnrate

        var before = EvaluateShort();
        Config.Optimizer.Init();
        Context.FullReset();
        foreach (var epochIndex in ..Config.EpochCount)
        {
            var epoch = Config.GetEpoch();
            var batchCount = 0;

            //if(Config.DumpEpochEvaluation) CallEvaluate();

            foreach (var batch in epoch)
            {

                var evaluation = Context.TrainAndEvaluate(batch);
                if (Config.DumpBatchEvaluation && batchCount % Config.DumpEvaluationAfterBatches == 0)
                {
                    Config.EvaluationCallback!.Invoke(new() { Context = GetContext(), Result = evaluation });
                }
                batchCount++;
                Config.Optimizer.OnBatchCompleted();
            }

            Config.Optimizer.OnEpochCompleted();

            TrainingEvaluationContext GetContext() => new()
            {
                CurrentBatch = batchCount,
                MaxBatch = epoch.BatchCount,
                CurrentEpoch = epochIndex + 1,
                MaxEpoch = Config.EpochCount,
                LearnRate = Config.Optimizer.LearningRate,

            };
        }

        return new()
        {
            EpochCount = Config.EpochCount,
            Before = before,
            After = EvaluateShort(),
        };
    }

    public ModelEvaluation EvaluateShort(TrainingEvaluationContext context) => new()
    {
        Context = context,
        Result = EvaluateShort(),
    };
    public ModelEvaluationResult EvaluateShort() => new()
    {
        TrainingSetResult = Evaluate(Config.GetRandomTrainingBatch()),
        TestSetResult = Evaluate(Config.GetRandomTestBatch()),
    };
    public DataSetEvaluationResult Evaluate(Batch<TInput, TOutput> batch)
    {
        int correctCounter = 0;
        Number totalCost = 0;
        int totalCounter = 0;
        foreach (var entry in batch)
        {
            totalCounter++;
            var output = Network.Process(entry.Input);

            if (output.Equals(entry.Expected))
            {
                correctCounter++;
            }

            totalCost += Config.CostFunction.TotalCost(Network.LastOutputWeights, Config.OutputResolver.Expected(entry.Expected));
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }
}