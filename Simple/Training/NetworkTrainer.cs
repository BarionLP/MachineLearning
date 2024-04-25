using System.Text;
using Simple.Network;

namespace Simple.Training;

public sealed class NetworkTrainer<TInput, TOutput>(TrainingConfig<TInput, TOutput> config, RecordingNetwork<TInput, TOutput> network) where TInput : notnull where TOutput : notnull{
    public TrainingConfig<TInput, TOutput> Config { get; } = config;
    public RecordingNetwork<TInput, TOutput> Network { get; } = network;
    internal NetworkTrainingContext<TInput, TOutput> Context { get; } = new(network, config.CostFunction, config.OutputResolver);

    public NetworkTrainingResult Train() {
        var before = Evaluate();

        var learnRate = Config.LearnRate;
        if(Config.DumpEvaluationAfterIterations > 0) Console.WriteLine($"Starting Training (learnRate: {learnRate:P}): {before.DumpShort()}");
        foreach(var iteration in ..Config.Iterations) {
            Context.Learn(Config.GetNextTrainingBatch(), learnRate, Config.Regularization, Config.Momentum);
            if(iteration > 0 && Config.DumpEvaluationAfterIterations > 0 && iteration % Config.DumpEvaluationAfterIterations == 0) {
                Console.WriteLine($"{Evaluate().DumpShort()} after {iteration} iterations (learnRate: {learnRate:P})");
            }
            learnRate *= Config.LearnRateMultiplier;
        }

        var after = Evaluate();
        return new() { 
            IterationCount = Config.Iterations,
            Before = before, 
            After = after, 
        };
    }

    public NetworkEvaluationResult Evaluate() => new() {
        TrainingSetResult = Evaluate(Config.GetNextTrainingBatch()),
        TestSetResult = Evaluate(Config.GetNextTestBatch()),
    };
    public DataSetEvaluationResult Evaluate(IEnumerable<DataPoint<TInput, TOutput>> dataSet) {
        int correctCounter = 0;
        Number totalCost = 0;
        int totalCounter = 0;
        foreach(var entry in dataSet) {
            totalCounter++;
            var output = Network.Process(entry.Input);

            if(output.Equals(entry.Expected)) {
                correctCounter++;
            }
            
            totalCost += Config.CostFunction.TotalCost(Network.LastOutputWeights, Config.OutputResolver.Expected(entry.Expected));
        }

        return new() {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }
}

public sealed class DataSetEvaluationResult {
    public required int TotalCount { get; init; }
    public required int CorrectCount { get; init; }
    public float CorrectPercentage => (float) CorrectCount / TotalCount;
    public required Number TotalCost { get; init; }
    public Number AverageCost => TotalCost / TotalCount;
}

public sealed class NetworkEvaluationResult {
    public required DataSetEvaluationResult TrainingSetResult { get; init; }
    public required DataSetEvaluationResult TestSetResult { get; init; }
    public void DumpCorrectPrecentages(StringBuilder sb) => sb.Append($"{TrainingSetResult.CorrectPercentage:P} | {TestSetResult.CorrectPercentage:P}");
    public string DumpShort() => $"Correct: {TrainingSetResult.CorrectPercentage:P} | {TestSetResult.CorrectPercentage:P}";
}

public sealed class NetworkTrainingResult {
    public required int IterationCount { get; init; }
    public required NetworkEvaluationResult Before { get; init; }
    public required NetworkEvaluationResult After { get; init; }
    public string DumpShort(){
        var sb = new StringBuilder();
        sb.Append("Training Results: ");
        Before.DumpCorrectPrecentages(sb);
        sb.Append(" -> ");
        After.DumpCorrectPrecentages(sb);
        return sb.ToString();
    }
}
