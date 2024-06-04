namespace MachineLearning.Samples;

public class SimpleLM
{
    public static int ContextSize = 46;
    public static SimpleNetwork<string, char, RecordingLayer> GetModel()
    {
        var initializer = new XavierInitializer(new Random(69));
        return NetworkBuilder.Recorded<string, char>(ContextSize * 8)
            .SetDefaultActivationMethod(SigmoidActivation.Instance)
            .SetEmbedder(new StringEmbedder(ContextSize))
            .AddLayer(512, initializer)
            .AddLayer(256, initializer)
            .AddLayer(64, initializer)
            .AddLayer(8, initializer)
            .Build();
    }

    public static TrainingConfig<string, char> GetTrainingConfig()
    {
        var dataSet = SimpleSentencesDataSource.GenerateData(ContextSize).ToArray();
        new Random(128).Shuffle(dataSet);

        var trainingSetSize = (int)(dataSet.Length * 0.9);
        return new TrainingConfig<string, char>()
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
            ShuffleTrainingSetPerEpoch = true,
        };
    }

    public static SimpleNetwork<string, char, RecordingLayer> TrainDefault() => TrainDefault(GetModel());
    public static SimpleNetwork<string, char, RecordingLayer> TrainDefault(SimpleNetwork<string, char, RecordingLayer> model)
    {
        var config = GetTrainingConfig();

        var trainer = new NetworkTrainer<string, char>(config, model);

        trainer.Train();

        var data = "They ".ToLowerInvariant();
        Console.Write(data);
        char prediction;
        do
        {
            prediction = model.Process(data);
            data += prediction;
            Console.Write(prediction);
        } while (prediction != '.' && data.Length < ContextSize);
        Console.WriteLine();

        return model;
    }
}
