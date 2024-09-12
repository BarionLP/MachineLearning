using MachineLearning.Samples.Language;
using MachineLearning.Serialization;

namespace MachineLearning.Samples.Logic;

public sealed class LogicModel : ISample<string, char>
{
    public const int CONTEXT_SIZE = 24;
    public const string TOKENS = "0123456789+-*/^\0";

    public static IEmbedder<string, char> Embedder { get; } = new BinaryStringEmbedder(CONTEXT_SIZE, TOKENS, false);
    public static IOutputResolver<char> OutputResolver { get; } = new CharOutputResolver(TOKENS);
    public static ModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("logic.nnw"));
    public static EmbeddedModel<string, char> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return new ModelBuilder(CONTEXT_SIZE * 8)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .AddLayer(512, initializer)
            .AddLayer(512, initializer)
            .AddLayer(256, initializer)
            .AddLayer(TOKENS.Length, initializer, SoftMaxActivation.Instance)
            .Build(Embedder);
    }

    public static TrainingConfig<string, char> DefaultTrainingConfig(Random? random = null) => new()
    {
        TrainingSet = LogicalStatementSource.GenerateAdditionStatements(1024, random).ToArray(),
        TestSet = LogicalStatementSource.GenerateAdditionStatements(128 * 2, random).ToArray(),

        EpochCount = 32,
        BatchCount = 512 * 3,

        Optimizer = new AdamWOptimizer
        {
            LearningRate = 0.1,
            CostFunction = CrossEntropyLoss.Instance,
        },

        OutputResolver = new CharOutputResolver(TOKENS),

        DumpEvaluationAfterBatches = 32,
        EvaluationCallback = result => Console.WriteLine(result.Dump()),

        RandomSource = random ?? Random.Shared,
    };

    public static void OpenChat(EmbeddedModel<string, char> model) {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrEmpty(input))
            {
                return;
            }
            Console.CursorTop--;
            Console.CursorLeft = 0;
            Console.WriteLine(Complete(input, model));
        } while (true);
    }

    public static string Complete(string input, EmbeddedModel<string, char> model)
    {
        while (true)
        {
            var (prediction, _) = model.Process(input);
            input += prediction;
            if (prediction == '\0')
            {
                break;
            }
        }

        return input;
    }

    public static EmbeddedModel<string, char> TrainDefault(EmbeddedModel<string, char>? model = null, TrainingConfig<string, char>? trainingConfig = null, Random? random = null)
    {
        var trainer = ModelTrainer.Legacy(model ?? CreateModel(random), trainingConfig ?? DefaultTrainingConfig(random));
        trainer.TrainConsole();
        return trainer.Model;
    }

    public static IEnumerable<DataEntry<string, char>> GetTrainingSet(Random? random = null) => LogicalStatementSource.GenerateAdditionStatements(1024, random);
}
