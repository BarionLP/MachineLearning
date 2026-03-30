using System.IO;
using ML.Core.Converters;
using ML.Core.Data.Training;
using ML.Core.Evaluation.Cost;
using ML.Core.Modules;
using ML.Core.Modules.Activations;
using ML.Core.Training;

namespace ML.Runner.Samples.Language;

public static class SLM4
{
    public static readonly HashSet<string> WORD_TOKENS = ["the", "and", "to", "of", "in", "is", "for", "that", "you", "with", "on", "it", "are", "as", "this", "be", "your", "or", "have", "at", "was", "from", "we", "by", "will", "not", "can", "an", "but", "all", "they", "if", "has", "our", "my", "more", "their", "one", "so", "he", "about", "which", "when", "what", "also", "out", "his", "up", "there", "time", "new", "do", "who", "like", "some", "other", "been", "just", "get", "how", "her", "would", "had", "them", "were", "any", "no", "these", "into", "me", "than", "people", "its", "make", "most", "only", "may", "she", "us", "first", "over", "use", "work", "very", "after", "well", "then", "now", "many", "need", "even", "through", "way", "two", "good", "best", "because", "see", "years", "know", "where", "day", "should", "much", "could", "such", "great", "here", "while", "take", "help", "home", "said", "back", "want", "it's", "being", "before", "year", "those", "find", "each", "made", "right", "used", "life", "go", "world", "free", "information", "business", "really", "every", "love", "think", "own", "both", "still", "around", "him", "last", "going", "off", "same", "different", "look", "place", "part", "between", "too", "did", "down", "service", "am", "during", "does", "since", "using", "high", "things", "company", "always", "another", "few", "set", "little", "available", "long", "services", "without", "online", "don't", "system", "family", "experience", "something", "come", "data", "next", "school", "why", "better", "sure", "under", "give", "however", "must", "including", "support", "can't"];
    public const string SYMBOLS = "\0 ?!\"#$%&'()*+,-./0123456789:;=?_abcdefghijklmnopqrstuvwxyz|ßäöü€";
    public const int CONTEXT_SIZE = int.MaxValue;
    public static StringTokenizer Tokenizer { get; } = new(WORD_TOKENS, SYMBOLS, [("“", "\""), ("”", "\""), ("\n", " "), ("–", "-"), ("—", "-"), ("’", "'"), ("it’s", "it's"), ("don’t", "don't"), ("can’t", "can't")], StringComparer.InvariantCultureIgnoreCase);
    public static FileInfo ModelFile { get; } = AssetManager.GetModelFile("slm4");

    public static EmbeddedModule<int[], Matrix, int> CreateAndInitModel(Random random)
    {
        const int EMBEDDING_SIZE = 92;
        var innerModel = new SequenceModule<Matrix>
        {
            Inner = [
                new LinearMatrixModule(EMBEDDING_SIZE, EMBEDDING_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(EMBEDDING_SIZE, EMBEDDING_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(EMBEDDING_SIZE, EMBEDDING_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(EMBEDDING_SIZE, EMBEDDING_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(EMBEDDING_SIZE, EMBEDDING_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(EMBEDDING_SIZE, EMBEDDING_SIZE),
                LeakyReLUActivation.Instance,
            ],
        };

        var input = new IndexEmbeddingModule(Tokenizer.TokenCount, EMBEDDING_SIZE);
        var output = new IndexUnembeddingModule(new(Tokenizer.TokenCount, weightedRandom: true, random), EMBEDDING_SIZE);

        NumericsInitializer.XavierUniform(input.EmbeddingMatrix, random);
        NumericsInitializer.XavierUniform(output.UnembeddingMatrix, random);

        var initer = new LinearModule.KaimingInitializer(LeakyReLUActivation.Instance) { Random = random };
        foreach (var linear in innerModel.Inner.OfType<LinearMatrixModule>())
        {
            initer.Init(linear);
        }

        return new EmbeddedModule<int[], Matrix, int>
        {
            Input = input,
            Hidden = innerModel,
            Output = output,
        };
    }

    public static void Run(Random random)
    {
        var trainingConfig = new TrainingConfig
        {
            EpochCount = 1,
            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.0002f,
            },

            EvaluationCallbackAfterBatches = 8,
            EvaluationCallback = evaluation => Console.WriteLine(evaluation),
            Threading = ThreadingMode.Full, // half seems to be faster than full
        };

        var model = ModuleSerializer.Read<EmbeddedModule<int[], Matrix, int>>(ModelFile);
        // var model = CreateAndInitModel(random);

        // var trainingSource = GetTrainingSource(random);
        using var trainingSource = GetC4DataSet();

        // IndexUnembeddingModule always output logits (for now)
        var trainer = new EmbeddedModuleTrainer<int[], Matrix, int>(model, trainingConfig)
        {
            CostFunction = CrossEntropyCostFromLogits.Instance,
            TrainingData = trainingSource,
        };

        ((IndexUnembeddingModule)model.Output).OuputLogits = true;
        trainer.TrainConsole();
        Console.WriteLine(trainingSource.GetState());

        trainingSource.ToString();

        trainer.DataPool.Clear();

        ModuleSerializer.Write(model, ModelFile);

        ((IndexUnembeddingModule)model.Output).OuputLogits = false;
        LMHelper.StartChat(model, CONTEXT_SIZE, Tokenizer);
    }

    public static TrainingDataSource<TrainingEntry<int[], Matrix, int>> GetTrainingSource(Random random)
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

        return new(lines.Tokenize(Tokenizer).Select(input => (input.ToArray(), endToken)).ToTrainingDataMatrix(Tokenizer.TokenCount))
        {
            BatchCount = 256,
            Random = random ?? Random.Shared,
        };
    }

    public static C4DataSet GetC4DataSet()
    {
        return new(Tokenizer, initalFile: 0) { BatchSize = 512 };
    }
}
