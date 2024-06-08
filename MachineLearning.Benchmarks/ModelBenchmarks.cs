using MachineLearning.Data.Entry;
using MachineLearning.Samples;
using MachineLearning.Samples.MNIST;
using MachineLearning.Training;

namespace MachineLearning.Benchmarks;

[MemoryDiagnoser()]
public class ModelBenchmarks
{
    private IEnumerable<DataEntry<double[], int>> dataSet = [];
    private ModelTrainer<double[], int> trainer = null!;

    [GlobalSetup]
    public void Setup(){
        var source = new MNISTDataSource(AssetManager.MNISTArchive);
        dataSet = source.TrainingSet.Take(256).ToArray();
        trainer = new ModelTrainer<double[], int>(MNISTModel.GetModel(), MNISTModel.GetTrainingConfig());
    }

    [Benchmark]
    public void Benchmark(){
        trainer.Context.FullReset();
        trainer.Context.TrainAndEvaluate(dataSet, true);
    }
}
