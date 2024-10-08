using MachineLearning.Model.Layer;

namespace MachineLearning.Samples.Language;

public sealed class SLM3Mini
{
    public const string TOKENS = " %'(),-.0123456789:=abcdefghijklmnopqrstuvwxyz\0";
    public const int CONTEXT_SIZE = 64;

    public static IOutputResolver<char> OutputResolver { get; } = new CharOutputResolver(TOKENS);
    public static GenericModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("sentences_3_mini.gmw"));
    public static FeedForwardModel<string, char> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return AdvancedModelBuilder
            .Create<StringEmbeddingLayer, string>(new StringEmbeddingLayer(TOKENS, CONTEXT_SIZE, 12), new StringEmbeddingLayer.Initializer(random))
                .SetDefaultActivationFunction(LeakyReLUActivation.Instance)
                .AddLayer(1024, initializer)
                .AddLayer(512, initializer)
                .AddLayer(256, initializer)
                .AddLayer(TOKENS.Length, new XavierInitializer(random), SoftMaxActivation.Instance)
            .AddOutputLayer(new TokenOutputLayer(TOKENS, true, random));
    }

    public static IEmbeddedModel<string, char> TrainDefault(IEmbeddedModel<string, char>? model = null, TrainingConfig<string, char>? config = null, Random? random = null)
    {
        model ??= CreateModel(random);

        var trainer = ModelTrainer.Generic(model, config ?? DefaultTrainingConfig());
        trainer.TrainConsole();
        Serializer.Save(model).Resolve(
            () => Console.WriteLine("Model saved!"),
            flag => Console.WriteLine($"Error saving model: {flag}")
        );
        LMHelper.StartChat(model, CONTEXT_SIZE);
        return model;
    }

    public static TrainingConfig<string, char> DefaultTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;

        var dataSet = GetTrainingSet().ToArray();
        random.Shuffle(dataSet);

        var trainingSetSize = (int)(dataSet.Length * 0.9);

        return new TrainingConfig<string, char>()
        {
            TrainingSet = dataSet,
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 32,
            BatchCount = 256,

            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.02,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = OutputResolver,

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 64,

            RandomSource = random,
        };
    }

    public static IEnumerable<DataEntry<string, char>> GetTrainingSet(Random? random = null)
    {
        Console.WriteLine("Analyzing Trainings Data...");
        var lines = LanguageDataSource.GetLines(AssetManager.Sentences).ToArray();
        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)} chars");
        var tokensUsedBySource = new string(lines.SelectMany(s => s).Distinct().Order().ToArray());
        Console.WriteLine($"Source uses '{tokensUsedBySource}'");
        tokensUsedBySource.ForEach(t => OutputResolver.Expected(t));

        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));
        return lines.InContextSize(CONTEXT_SIZE).ExpandPerChar();
    }
}
