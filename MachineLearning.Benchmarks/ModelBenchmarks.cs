using MachineLearning.Data.Entry;
using MachineLearning.Samples;
using MachineLearning.Samples.MNIST;
using MachineLearning.Training;

namespace MachineLearning.Benchmarks;

[MemoryDiagnoser]
public class ModelBenchmarks
{
    private IEnumerable<DataEntry<double[], int>> dataSet = [];
    private LegacyModelTrainer<double[], int> trainer = null!;

    [GlobalSetup]
    public void Setup(){
        var source = new MNISTDataSource(AssetManager.MNISTArchive);
        dataSet = source.TrainingSet.Take(256).ToArray();
        trainer = new LegacyModelTrainer<double[], int>(MNISTModel.CreateModel(), MNISTModel.GetTrainingConfig());
    }

    [Benchmark]
    public void Benchmark(){
        trainer.Context.FullReset();
        trainer.Context.TrainAndEvaluate(dataSet, true);
    }
}
