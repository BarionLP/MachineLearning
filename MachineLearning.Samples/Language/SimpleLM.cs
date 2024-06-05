using ModelDefinition = MachineLearning.Model.SimpleNetwork<string, char, MachineLearning.Model.Layer.RecordingLayer>;

namespace MachineLearning.Samples.Language;

public static class SimpleLM
{
    public const int ContextSize = 256;
    public static ModelDefinition GetModel(Random? random = null)
    {
        var initializer = new XavierInitializer(random);
        return NetworkBuilder.Recorded<string, char>(ContextSize * 8)
            .SetDefaultActivationMethod(SigmoidActivation.Instance)
            .SetEmbedder(new StringEmbedder(ContextSize))
            .AddLayer(1024, initializer)
            .AddLayer(256, initializer) //512
            .AddLayer(LanguageDataSource.TOKENS.Length, builder => builder.Initialize(initializer).SetActivationMethod(SoftmaxActivation.Instance))
            .Build();
    }

    public static TrainingConfig<string, char> GetTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;
        var dataSet = LanguageDataSource.SpeechData(ContextSize).ToArray();
        random.Shuffle(dataSet);

        var trainingSetSize = (int) (dataSet.Length * 0.9);
        return new TrainingConfig<string, char>()
        {
            TrainingSet = dataSet.Take(trainingSetSize).ToArray(),
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 4,
            BatchCount = 256,

            Optimizer = new AdamOptimizerConfig
            {
                LearningRate = 0.09,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new CharOutputResolver(),

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 16,

            RandomSource = random,
            ShuffleTrainingSetPerEpoch = true,
        };
    }

    public static ModelDefinition TrainDefault(Random? random = null) => TrainDefault(GetModel(random ?? Random.Shared), random);
    public static ModelDefinition TrainDefault(ModelDefinition model, Random? random = null)
    {
        var config = GetTrainingConfig(random ?? Random.Shared);
        var trainer = ModelTrainer.Create(model, config);

        trainer.Train();

        Generate("deutschland ", model);

        return model;
    }

    public static void Generate(string input, ModelDefinition model)
    {
        input = input.ToLowerInvariant();
        Console.Write(input);
        char prediction;
        do
        {
            prediction = model.Process(input);
            input += prediction;
            Console.Write(prediction);
        } while(prediction != '.' && input.Length < ContextSize);
        Console.WriteLine();
    }

    public static void StartChat(ModelDefinition model)
    {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            Generate(input, model);
        } while(!string.IsNullOrEmpty(input));
    }
}
