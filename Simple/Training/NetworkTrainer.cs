using Simple.Network;

namespace Simple.Training;

public sealed class NetworkTrainer(TrainingConfig config, RecordingNetwork network) {
    public TrainingConfig Config { get; } = config;
    public RecordingNetwork Network { get; } = network;
    internal NetworkTrainingContext Context { get; } = new(network, config.CostFunction);

    public NetworkTrainingResult Train() {
        var before = Evaluate();

        var learnRate = Config.LearnRate;
        Console.WriteLine($"Iteration 0 (learnRate: {learnRate:P}): {before.DumpShort()}");
        foreach(var iteration in ..Config.Iterations) {
            Context.Learn(Config.GetNextTrainingBatch(), learnRate);
            learnRate *= Config.LearnRateMultiplier;
            if(iteration > 0 && iteration % Config.DumpEvaluationAfterIterations == 0) {
                Console.WriteLine($"Iteration {iteration} (learnRate: {learnRate:P}): {Evaluate().DumpShort()}");
            }
        }

        var after = Evaluate();
        return new() { 
            IterationCount = Config.Iterations,
            Before = before, 
            After = after, 
        };
    }

    public NetworkEvaluationResult Evaluate() {
        return new() {
            TrainingSetResult = Evaluate(Config.TrainingSet),
            TestSetResult = Evaluate(Config.TestSet),
        };
    }
    public DataSetEvaluationResult Evaluate(IEnumerable<DataPoint> dataSet) {
        int correctCounter = 0;
        Number totalCost = 0;
        int totalCounter = 0;
        foreach(var entry in dataSet) {
            totalCounter++;
            var output = Network.Process(entry.Input);

            //TODO: define for all model types (using embedder)
            if(output[0] > output[1] && entry.Expected[0] > entry.Expected[1]) {
                correctCounter++;
            } else if(output[0] < output[1] && entry.Expected[0] < entry.Expected[1]) {
                correctCounter++;
            }
            totalCost += Config.CostFunction.TotalCost(output, entry.Expected);
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
    public string DumpShort() => $"Correct: {TrainingSetResult.CorrectPercentage:P} | {TestSetResult.CorrectPercentage:P}";
}

public sealed class NetworkTrainingResult {
    public required int IterationCount { get; init; }
    public required NetworkEvaluationResult Before { get; init; }
    public required NetworkEvaluationResult After { get; init; }
}
