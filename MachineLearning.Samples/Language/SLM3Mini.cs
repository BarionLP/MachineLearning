using MachineLearning.Model.Layer;

namespace MachineLearning.Samples.Language;

public sealed class SLM3Mini
{
    public const string TOKENS = " %'(),-.0123456789:=abcdefghijklmnopqrstuvwxyz\0";
    public const int CONTEXT_SIZE = 64;

    public static IOutputResolver<char> OutputResolver { get; } = new CharOutputResolver(TOKENS);
    public static GenericModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("sentences_3_mini.gmw"));
    public static CharTokenizer Tokenizer { get; } = new(TOKENS);
    public static FeedForwardModel<int[], char> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return AdvancedModelBuilder
            .Create(new EncodedEmbeddingLayer(TOKENS.Length, CONTEXT_SIZE))
                .SetDefaultActivationFunction(LeakyReLUActivation.Instance)
                .AddLayer(1024, initializer)
                .AddLayer(512, initializer)
                .AddLayer(256, initializer)
                .AddLayer(TOKENS.Length, new XavierInitializer(random), SoftMaxActivation.Instance)
            .AddOutputLayer(new TokenOutputLayer(TOKENS, true, random));
    }

    public static IEmbeddedModel<int[], char> TrainDefault(IEmbeddedModel<int[], char>? model = null, TrainingConfig<int[], char>? config = null, Random? random = null)
    {
        model ??= CreateModel(random);

        var trainer = ModelTrainer.Generic(model, config ?? DefaultTrainingConfig());
        trainer.TrainConsole();
        Serializer.Save(model).Consume(
            () => Console.WriteLine("Model saved!"),
            flag => Console.WriteLine($"Error saving model: {flag}")
        );
        LMHelper.StartChat(model, CONTEXT_SIZE, Tokenizer);
        return model;
    }

    public static TrainingConfig<int[], char> DefaultTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;

        var dataSet = GetTrainingSet().ToArray();
        random.Shuffle(dataSet);

        var trainingSetSize = (int)(dataSet.Length * 0.9);

        return new()
        {
            TrainingSet = dataSet,
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 32,
            BatchCount = 256,

            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.02f,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = OutputResolver,

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 64,

            RandomSource = random,
        };
    }

    public static IEnumerable<DataEntry<int[], char>> GetTrainingSet(Random? random = null)
    {
        Console.WriteLine("Analyzing Trainings Data...");
        var lines = LanguageDataSource.GetLines(AssetManager.Sentences).ToArray();
        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)} chars");
        var tokensUsedBySource = new string([.. lines.SelectMany(s => s).Distinct().Order()]);
        Console.WriteLine($"Source uses '{tokensUsedBySource}'");
        tokensUsedBySource.Consume(t => OutputResolver.Expected(t));

        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));
        return lines.InContextSize(CONTEXT_SIZE).ExpandPerChar().Select(d => new DataEntry<int[], char>(Tokenizer.Tokenize(d.Input), d.Expected));
    }
}

public sealed class CharTokenizer(string tokens)
{
    private readonly string tokens = tokens;

    public int[] Tokenize(string input)
    {
        var result = new int[input.Length];
        foreach (var i in ..input.Length)
        {
            result[i] = Tokenize(input[i]);
        }
        return result;
    }

    public int Tokenize(char input) => tokens.IndexOf(input);
    public char Reverse(int input) => tokens[input];
}