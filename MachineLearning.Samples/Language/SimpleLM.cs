using ModelDefinition = MachineLearning.Model.EmbeddedModel<string, char>;

namespace MachineLearning.Samples.Language;

public sealed class SimpleLM : ISample<string, char>
{
    public const int CONTEXT_SIZE = 256 + 64;
    public static ModelDefinition CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return new ModelBuilder(CONTEXT_SIZE * 8)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .AddLayer(2048, initializer)
            .AddLayer(512, initializer)
            .AddLayer(128, initializer)
            .AddLayer(LanguageDataSource.TOKENS.Length, builder => builder.Initialize(new XavierInitializer(random)).SetActivationMethod(SoftmaxActivation.Instance))
            .Build(new BinaryStringEmbedder(CONTEXT_SIZE, LanguageDataSource.TOKENS, true));
    }

    public static TrainingConfig<string, char> DefaultTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;
        var dataSet = GetTrainingSet().ToArray();
        random.Shuffle(dataSet);

        var trainingSetSize = (int)(dataSet.Length * 0.9);
        return new TrainingConfig<string, char>()
        {
            TrainingSet = dataSet.Take(trainingSetSize).ToArray(),
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 8,
            BatchCount = 256,

            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.01,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new CharOutputResolver(LanguageDataSource.TOKENS),

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 32,

            RandomSource = random,
            ShuffleTrainingSetPerEpoch = true,
        };
    }

    public static ModelDefinition TrainDefault(ModelDefinition? model = null, TrainingConfig<string, char>? config = null, Random? random = null)
    {
        model ??= CreateModel(random);
        config ??= DefaultTrainingConfig(random);

        var trainer = ModelTrainer.Create(model, config);
        trainer.TrainConsole();

        return model;
    }

    public static IEnumerable<DataEntry<string, char>> GetTrainingSet(Random? random = null) => LanguageDataSource.GetLines(AssetManager.Speech).InContextSize(CONTEXT_SIZE).ExpandPerChar();
}
