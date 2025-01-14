using MachineLearning.Data.Entry;
using MachineLearning.Samples.MNIST;
using MachineLearning.Training;

namespace MachineLearning.Benchmarks;

[MemoryDiagnoser]
public class ModelBenchmarks
{
    private IEnumerable<TrainingData<double[], int>> dataSet = [];
    private EmbeddedModelTrainer<double[], int> trainer = null!;

    [GlobalSetup]
    public void Setup()
    {
        var source = MNISTModel.GetTrainingSet();
        dataSet = [.. source.GetBatches().First().Cast<TrainingData<double[], int>>()];
        trainer = new EmbeddedModelTrainer<double[], int>(MNISTModel.CreateModel(), MNISTModel.DefaultTrainingConfig(), source);
    }

    [Benchmark]
    public void Benchmark()
    {
        // trainer.FullReset();
        trainer.TrainAndEvaluate(dataSet);
    }
}
