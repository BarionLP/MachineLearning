using MachineLearning.Data;
using MachineLearning.Model.Layer;

namespace MachineLearning.Samples.Language;

public sealed class SLM3Mini : ISample<int[], int>
{
    public const string TOKENS = " %'(),-.0123456789:=abcdefghijklmnopqrstuvwxyz\0";
    public const int CONTEXT_SIZE = 64;

    public static ModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("slm3_mini.gmw"));
    public static EmbeddedModel<int[], int> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return AdvancedModelBuilder
            .Create(new EncodedEmbeddingLayer(TOKENS.Length, CONTEXT_SIZE))
                .DefaultActivation(LeakyReLUActivation.Instance)
                .AddLayer(1024, initializer)
                .AddLayer(512, initializer)
                .AddLayer(256, initializer)
                .AddLayer(TOKENS.Length, new XavierInitializer(random), SoftMaxActivation.Instance)
            .AddOutputLayer(new TokenOutputLayer(TOKENS.Length, true, random));
    }

    public static EmbeddedModel<int[], int> TrainDefault(EmbeddedModel<int[], int>? model = null, TrainingConfig? config = null, Random? random = null)
    {
        model ??= Serializer.Load<EmbeddedModel<int[], int>>().Or(error =>
        {
            Console.WriteLine("No existing model found! Creating new!");
            return CreateModel(random);
        });

        var trainer = ModelTrainer.Create(model, config ?? DefaultTrainingConfig(), GetTrainingSet());
        trainer.TrainConsole();
        Serializer.Save(model).Consume(
            () => Console.WriteLine("Model saved!"),
            error => Console.WriteLine($"Error saving model: {error.Message}")
        );
        LMHelper.StartChat(model, CONTEXT_SIZE, TOKENS);
        return model;
    }

    public static TrainingConfig DefaultTrainingConfig(Random? random = null) => new()
    {
        EpochCount = 32,

        Optimizer = new AdamOptimizer
        {
            LearningRate = 0.02f,
            CostFunction = CrossEntropyLoss.Instance,
        },

        EvaluationCallback = result => Console.WriteLine(result.Dump()),
        DumpEvaluationAfterBatches = 32,

        RandomSource = random ?? Random.Shared,
    };

    public static ITrainingSet GetTrainingSet(Random? random = null)
    {
        Console.WriteLine("Analyzing Trainings Data...");
        var lines = LanguageDataHelper.GetLines(AssetManager.Sentences).ToArray();
        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)} chars");
        var tokensUsedBySource = new string([.. lines.SelectMany(s => s).Distinct().Order()]);
        Console.WriteLine($"Source uses '{tokensUsedBySource}'");

        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));
        var entries = lines.Select(s => s.EndsWith('\0') ? s : s + '\0').InContextSize(CONTEXT_SIZE).ExpandPerChar();
        return new PredefinedTrainingSet(entries.ToTrainingData(TOKENS))
        {
            BatchCount = 256,
            Random = random ?? Random.Shared,
        };
    }
}
