using System.IO;
using ML.Core.Attributes;
using ML.Core.Converters;
using ML.Core.Data.Training;
using ML.Core.Evaluation.Cost;
using ML.Core.Modules;
using ML.Core.Modules.Activations;
using ML.Core.Modules.Builder;
using ML.Core.Training;

namespace ML.Runner.Samples.Language;

public static class SLM3
{
    public static readonly HashSet<string> WORD_TOKENS = ["the", "and", "to", "of", "in", "is", "for", "that", "you", "with", "on", "it", "are", "as", "this", "be", "your", "or", "have", "at", "was", "from", "we", "by", "will", "not", "can", "an", "but", "all", "they", "if", "has", "our", "my", "more", "their", "one", "so", "he", "about", "which", "when", "what", "also", "out", "his", "up", "there", "time", "new", "do", "who", "like", "some", "other", "been", "just", "get", "how", "her", "would", "had", "them", "were", "any", "no", "these", "into", "me", "than", "people", "its", "make", "most", "only", "may", "she", "us", "first", "over", "use", "work", "very", "after", "well", "then", "now", "many", "need", "even", "through", "way", "two", "good", "best", "because", "see", "years", "know", "where", "day", "should", "much", "could", "such", "great", "here", "while", "take", "help", "home", "said", "back", "want", "it's", "being", "before", "year", "those", "find", "each", "made", "right", "used", "life", "go", "world", "free", "information", "business", "really", "every", "love", "think", "own", "both", "still", "around", "him", "last", "going", "off", "same", "different", "look", "place", "part", "between", "too", "did", "down", "service", "am", "during", "does", "since", "using", "high", "things", "company", "always", "another", "few", "set", "little", "available", "long", "services", "without", "online", "don't", "system", "family", "experience", "something", "come", "data", "next", "school", "why", "better", "sure", "under", "give", "however", "must", "including", "support", "can't"];
    public const string SYMBOLS = "\0 ?!\"#$%&'()*+,-./0123456789:;=?_abcdefghijklmnopqrstuvwxyz|ßäöü€";
    public const int CONTEXT_SIZE = 128;
    public const int EMBEDDING_SIZE = 48;
    public static StringTokenizer Tokenizer { get; } = new(WORD_TOKENS, SYMBOLS, [("“", "\""), ("”", "\""), ("\n", " "), ("–", "-"), ("—", "-"), ("’", "'"), ("it’s", "it's"), ("don’t", "don't"), ("can’t", "can't")]);
    public static FileInfo ModelFile { get; } = AssetManager.GetModelFile("slm3");

    public static EmbeddedModule<int[], Vector, int> CreateAndInitModel(Random random)
    {
        var innerModel = MultiLayerPerceptronBuilder.Create(CONTEXT_SIZE * EMBEDDING_SIZE)
            .AddLayer(1024 * 2, LeakyReLUActivation.Instance)
            .AddLayer(1024 * 2, LeakyReLUActivation.Instance)
            .AddLayer(1024, LeakyReLUActivation.Instance)
            .AddLayer(1024, LeakyReLUActivation.Instance)
            .AddLayer(Tokenizer.TokenCount, SoftMaxActivation.Instance)
        .BuildAndInit(random);

        var input = new IndexEmbeddingModule(Tokenizer.TokenCount, EMBEDDING_SIZE);

        NumericsInitializer.XavierUniform(input.EmbeddingMatrix, random);

        return new EmbeddedModule<int[], Vector, int>
        {
            Input = new MatrixToFixedVectorModule(CONTEXT_SIZE, EMBEDDING_SIZE, input),
            Hidden = innerModel,
            Output = new IndexOutputLayer(Tokenizer.TokenCount, weightedRandom: false, random),
        };
    }

    public static void Run(Random random)
    {
        var trainingConfig = new TrainingConfig
        {
            EpochCount = 1,
            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.0003f,
            },

            EvaluationCallbackAfterBatches = 8,
            EvaluationCallback = evaluation => Console.WriteLine(evaluation),
            Threading = ThreadingMode.Half, // half seems to be faster than full
        };

        var model = ModuleSerializer.Read<EmbeddedModule<int[], Vector, int>>(ModelFile);
        // var model = CreateAndInitModel(random);

        var trainingSource = GetTrainingSource(random);
        // using var trainingSource = GetC4DataSet();

        var trainer = new EmbeddedModuleTrainer<int[], Vector, int>(model, trainingConfig)
        {
            CostFunction = CrossEntropyCostFromProbabilities.Instance,
            TrainingData = trainingSource,
        };

        trainer.TrainConsole();

        trainer.DataPool.Clear();

        ModuleSerializer.Write(model, ModelFile);

        LMHelper.StartChat(model, CONTEXT_SIZE, Tokenizer);
    }

    public static TrainingDataSource<TrainingEntry<int[], Vector, int>> GetTrainingSource(Random random)
    {
        Console.WriteLine("Analyzing Trainings Data...");
        var lines = LanguageDataHelper.GetLines(AssetManager.Sentences).ToArray();
        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)} chars");
        var tokensUsedBySource = new string([.. lines.SelectMany(s => s).Distinct().Order()]);
        Console.WriteLine($"Source uses '{tokensUsedBySource}'");

        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));

        // var words = lines.SelectMany(l => l.Split([' ', '.', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        // var usages = words.CountBy(w => w).OrderByDescending(g => g.Value).Select(g => $"{g.Key}: {g.Value}");
        // Console.WriteLine(string.Join('\n', usages.Take(50)));
        var endToken = Tokenizer.TokenizeSingle("\0");

        return new(lines.Tokenize(Tokenizer).ExpandPerToken(endToken, CONTEXT_SIZE).ToTrainingData(Tokenizer.TokenCount))
        {
            BatchCount = 256,
            Random = random ?? Random.Shared,
        };
    }

    public static C4DataSet GetC4DataSet()
    {
        return new(Tokenizer, CONTEXT_SIZE) { BatchSize = 512 };
    }
}

[GeneratedModule(IncludeSerializer: true)]
internal sealed partial class MatrixToFixedVectorModule(int contextSize, int embeddingSize, IInputModule<int[], Matrix> inner) : IInputModule<int[], Vector>
{
    [Property] public int ContextSize { get; } = contextSize;
    [Property] public int EmbeddingSize { get; } = embeddingSize;

    [SubModule] public IInputModule<int[], Matrix> Inner { get; } = inner;

    public Vector Forward(int[] input, Snapshot snapshot)
    {
        snapshot.InputCount = input.Length;
        var matrix = Inner.Forward(input, snapshot.Inner);
        Debug.Assert(matrix.RowCount <= ContextSize);
        Debug.Assert(matrix.ColumnCount == EmbeddingSize);

        snapshot.Output.ResetZero();
        matrix.AsSpan().CopyTo(snapshot.Output.AsSpan()[^matrix.FlatCount..]);

        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        var matrixGradient = Matrix.Of(snapshot.InputCount, EmbeddingSize, outputGradient.Slice(0, snapshot.InputCount * EmbeddingSize));
        return Inner.Backward(matrixGradient, snapshot.Inner, gradients.Inner).Storage;
    }

    partial class Snapshot
    {
        public int InputCount { get; set; }
        public Vector Output { get; } = Vector.Create(module.ContextSize * module.EmbeddingSize);
    }

    [GeneratedAdam(typeof(MatrixToFixedVectorModule))]
    public sealed partial class Adam;
}