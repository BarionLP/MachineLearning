using MachineLearning.Domain.Activation;
using MachineLearning.Model;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Training;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;

namespace Simple;

public class SimpleLM
{
    void Code(){
        int contextSize = 46;
        var dataSet = SimpleSentencesDataSource.GenerateData(contextSize).ToArray();
        new Random(128).Shuffle(dataSet);

        var trainingSetSize = (int)(dataSet.Length * 0.9);
        var config = new TrainingConfig<string, char>()
        {
            TrainingSet = dataSet.Take(trainingSetSize).ToArray(),
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 8,
            BatchCount = trainingSetSize / (128 + 32),

            Optimizer = new AdamOptimizerConfig
            {
                LearningRate = 0.08,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new CharOutputResolver(),

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 16,

            RandomSource = new Random(42),
        };

        var setupRandom = new Random(69);
        var initializer = new XavierInitializer(setupRandom);
        var network = NetworkBuilder.Recorded<string, char>(contextSize * 8)
            .SetDefaultActivationMethod(SigmoidActivation.Instance)
            .SetEmbedder(new StringEmbedder(contextSize))
            .AddLayer(512, initializer)
            .AddLayer(256, initializer)
            .AddLayer(64, initializer)
            .AddLayer(32, initializer)
            .AddLayer(8, initializer)
            .Build();



        var data = "They ".ToLowerInvariant();
        Console.Write(data);
        char prediction;
        do
        {
            prediction = network.Process(data);
            data += prediction;
            Console.Write(prediction);
        } while (prediction != '.' && data.Length < 32);
        Console.WriteLine();
    }
}
