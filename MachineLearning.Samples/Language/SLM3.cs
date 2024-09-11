using System.Diagnostics;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;

namespace MachineLearning.Samples.Language;

public sealed class SLM3
{
    public const string TOKENS = " %'(),-.0123456789:=abcdefghijklmnopqrstuvwxyz\0";
    public const int CONTEXT_SIZE = 128;

    public static IOutputResolver<char> OutputResolver { get; } = new CharOutputResolver(TOKENS);
    public static ModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("sentence_2.nnw"));
    public static FeedForwardModel<string, char> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return AdvancedModelBuilder
            .Create<StringEmbeddingLayer, string>(new StringEmbeddingLayer(TOKENS, CONTEXT_SIZE, 12), new StringEmbeddingLayer.Initer(random))
                .SetDefaultActivationFunction(LeakyReLUActivation.Instance)
                .AddLayer(1024 * 2, initializer)
                .AddLayer(1024 * 2, initializer)
                .AddLayer(512 + 256, initializer)
                .AddLayer(512, initializer)
                .AddLayer(TOKENS.Length, new XavierInitializer(random), SoftMaxActivation.Instance)
            .AddOutputLayer(new TokenOutputLayer(TOKENS, true, random));
    }

    public static IEnumerable<DataEntry<string, char>> GetTrainingSet(Random? random = null)
    {
        Console.WriteLine("Analyzing Trainings Data...");
        var lines = LanguageDataSource.GetLines(AssetManager.Sentences).ToArray();
        //lines.ForEach(l => Embedder.Embed(l));
        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)} chars");
        var tokensUsedBySource = new string(lines.SelectMany(s => s).Distinct().Order().ToArray());
        Console.WriteLine($"Source uses '{tokensUsedBySource}'");
        tokensUsedBySource.ForEach(t => OutputResolver.Expected(t));

        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));
        return lines.InContextSize(CONTEXT_SIZE).ExpandPerChar();
    }
}

public sealed class TokenOutputLayer(string tokens, bool weightedRandom, Random? random = null) : IUnembeddingLayer<char>
{
    public string Tokens { get; } = tokens;
    public Random Random { get; } = random ?? Random.Shared;

    public int InputNodeCount => Tokens.Length;
    public uint ParameterCount => 0;

    public (char, Weight) Forward(Vector input)
    {
        Debug.Assert(input.Count == Tokens.Length);

        // temperature adjustments
        if (weightedRandom)
        {
            // input.PointwiseLogToSelf();
            // input.DivideToSelf(temperature);
            // input.SoftMaxToSelf();
        }

        var index = weightedRandom ? GetWeightedRandomIndex(input, Random) : input.MaximumIndex();
        return (Tokens[index], input[index]);

        static int GetWeightedRandomIndex(Vector weights, Random random)
        {
            var value = random.NextDouble();
            for (int i = 0; i < weights.Count; i++)
            {
                value -= weights[i];
                if (value < 0) return i;
            }
            return weights.Count - 1;
        }
    }
}


public sealed class StringEmbeddingLayer(string tokens, int contextSize, int embeddingSize) : IEmbeddingLayer<string>
{
    // init randomly with [-0.1; 0.1] or [-0.01; 0.01]
    public Matrix EmbeddingMatrix { get; } = Matrix.Create(tokens.Length, embeddingSize);
    public int OutputNodeCount { get; } = contextSize * embeddingSize;
    public string Tokens { get; } = tokens;
    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;
    public int ContextSize { get; } = contextSize;
    public uint ParameterCount => (uint)EmbeddingMatrix.FlatCount;

    public Vector Forward(string input)
    {
        var output = Vector.Create(OutputNodeCount);
        var outSpan = output.AsSpan();

        foreach (var i in ..input.Length)
        {
            var tokenIdx = Tokens.IndexOf(input[i]);
            if (tokenIdx < 0)
            {
                throw new ArgumentException($"Unkown token: '{input[i]}'");
            }

            EmbeddingMatrix.RowSpan(tokenIdx)
                .CopyTo(outSpan.Slice(i * EmbeddingSize, EmbeddingSize));
        }

        return output;
    }

    public sealed class Initer(Random? random = null) : ILayerInitializer<StringEmbeddingLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(StringEmbeddingLayer layer)
        {
            layer.EmbeddingMatrix.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.1));
        }
    }
}