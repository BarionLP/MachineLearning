using System.Diagnostics;
using MachineLearning.Model.Initialization;

namespace MachineLearning.Transformer;

public sealed class Model(ModelInfo info)
{
    public ModelInfo Info { get; } = info;
    public Embedder Embedder { get; } = new(info.ValidTokens, info.EmbeddingDimensions, info.Temperature);
    public AttentionBlock AttentionBlock { get; } = new(info);

    public void Initialize()
    {
        Info.Initializer.Initialize(Embedder.EmbeddingMatrix);
        Info.Initializer.Initialize(Embedder.UnembeddingMatrix);
        AttentionBlock.Initialize(Info.Initializer);
    }

    public int GetTokenCount()
    {
        return
            Info.TokenCount * Info.EmbeddingDimensions + // embedding
            AttentionBlock.GetWeightsCount() +
            Info.EmbeddingDimensions * Info.TokenCount  // unembedding
        ;
    }
}

public static class TransformerTest
{
    public static void Main()
    {
        //var model = new Model(new()
        //{
        //    ValidTokens = " 0123456789.",
        //    EmbeddingDimensions = 8,
        //    ContextSize = 128,
        //    KeyQueryDimensions = 12,
        //    AttentionBlockCount = 1,
        //    AttentionHeadCountPerBlock = 1,
        //    Temperature = 1.2,
        //    Initializer = new RandomInitializer(),
        //});

        //model.Initialize();

        //var testMessage = "3.1415";
        //DebugHelper.PrintTokenMatrix(testMessage, model.Embedder.Embedd(testMessage));
        //DebugHelper.PrintTokenMatrix(model.Info.ValidTokens, model.Embedder.EmbeddingMatrix);

        //Console.WriteLine(model.Embedder.Unembed(Vector.Of([1,1,1,1,1,1,1,1])));

        //model.AttentionBlock.Process(model.Embedder.Embedd(testMessage));

        var info = new ModelInfo()
        {
            ValidTokens = " 0123456789.",
            EmbeddingDimensions = 4,
            ContextSize = 4,
            KeyQueryDimensions = 2,
            AttentionBlockCount = 1,
            AttentionHeadCountPerBlock = 1,
            Temperature = 1.2,
            Initializer = new RandomInitializer(),
        };

        var head = new AttentionHead(info);

        head.Initialize(head.Info.Initializer);

        var pass = AttentionHead.HeadPass.Allocate(info);
        pass.input.MapToSelf(_ => 1);

        var result = head.GetEmbeddingDelta(pass);

        var loss = (pass.input, result).Map((input, result) => 0.5 * Math.Pow(result - input, 2));

        head.BackwardPass(loss, pass);
    }
}

public sealed class RandomInitializer : IInitializer
{
    public void Initialize(Span<Weight> vector)
    {
        SpanOperations.Map(vector, vector, s => Random.Shared.NextDouble());
    }
}

public static class DebugHelper
{
    public static void PrintTokenMatrix(string tokens, Matrix matrix)
    {
        Debug.Assert(tokens.Length == matrix.RowCount);
        Console.WriteLine();
        for (int i = 0; i < tokens.Length; i++)
        {
            Console.WriteLine($"'{tokens[i]}' => {matrix.RowRef(i)}");
        }
        Console.WriteLine();
    }
}