using MachineLearning.Data;
using MachineLearning.Model.Layer;
using ML.MultiLayerPerceptron;
using ML.MultiLayerPerceptron.Initialization;

namespace MachineLearning.Samples.Language;

public sealed class SLM3Mini : ISample<int[], int>
{
    public const int CONTEXT_SIZE = 64;

    public static CharTokenizer Tokenizer { get; } = new("\0 !%'(),-.0123456789:=?_abcdefghijklmnopqrstuvwxyzß");
    public static ModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("slm3_mini.gmw"));
    public static EmbeddedModel<int[], int> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return AdvancedModelBuilder
            .Create(new EncodedEmbeddingLayer(Tokenizer.TokenCount, CONTEXT_SIZE))
                .DefaultActivation(LeakyReLUActivation.Instance)
                .AddLayer(512 + 256, initializer)
                .AddLayer(512, initializer)
                .AddLayer(512, initializer)
                .AddLayer(Tokenizer.TokenCount, new XavierInitializer(random), SoftMaxActivation.Instance)
            .AddOutputLayer(new TokenOutputLayer(Tokenizer.TokenCount, true, random));
    }

    public static EmbeddedModel<int[], int> TrainDefault(EmbeddedModel<int[], int>? model = null, TrainingConfig? config = null, Random? random = null)
    {
        model ??= Serializer.Load<EmbeddedModel<int[], int>>().Or(error =>
        {
            Console.WriteLine("No existing model found! Creating new!");
            return CreateModel(random);
        });

        var trainer = new EmbeddedModelTrainer<int[], int>(model, config ?? DefaultTrainingConfig(), GetTrainingSet());
        trainer.TrainConsole();
        Serializer.Save(model).Consume(
            () => Console.WriteLine("Model saved!"),
            error => Console.WriteLine($"Error saving model: {error.Message}")
        );
        LMHelper.StartChat(model, CONTEXT_SIZE, Tokenizer);
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
        
        return new PredefinedTrainingSet(entries.ToTrainingData(Tokenizer))
        {
            BatchCount = 256,
            Random = random ?? Random.Shared,
        };
    }
}
