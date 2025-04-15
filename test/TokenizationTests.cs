using System.Collections;
using MachineLearning.Data;
using MachineLearning.Samples.Language;

namespace ML.Tests;

public sealed class TokenizationTests
{
    [Test]
    public async Task LanguageDataHelper_Test()
    {
        var pi = "314159265";
        var tokenizer = new TestTokenizer();

        var tokens = LanguageDataHelper.Tokenize([pi], tokenizer);
        await Assert.That(tokens.First()).IsEquivalentTo([3, 1, 4, 1, 5, 9, 2, 6, 5]);

        var slided = tokens.SlidingWindow(endToken: null, contextSize: 8);
        await Assert.That(slided.Count()).IsEqualTo(9);

        var slided2 = tokens.SlidingWindow(endToken: 10, contextSize: 8).Last();
        await Assert.That(slided2.Input).IsEquivalentTo([1, 4, 1, 5, 9, 2, 6, 5]);
        await Assert.That(slided2.Expected).IsEqualTo(10);

        var expected1 = slided.ToTrainingDataMatrix(tokenCount: 10, contextSize: 8, fillerToken: null).Last();
        await Assert.That(expected1.ExpectedWeights.AsSpan().SequenceEqual([
        //  0  1  2  3  4  5  6  7  8  9
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        ])).IsTrue();

        var expected2 = slided.ToTrainingDataMatrix(tokenCount: 10, contextSize: 8, fillerToken: null).First();
        await Assert.That(expected2.ExpectedWeights.AsSpan().SequenceEqual([
        //  0  1  2  3  4  5  6  7  8  9
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        ])).IsTrue();

        var expected3 = slided.ToTrainingDataMatrix(tokenCount: 11, contextSize: 8, fillerToken: 10).First();
        await Assert.That(expected3.ExpectedWeights.AsSpan().SequenceEqual([
        //  0  1  2  3  4  5  6  7  8  9 10
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])).IsTrue();


        var expected4 = slided.ToTrainingDataMatrix(tokenCount: 11, contextSize: 8, fillerToken: 10).Last();
        await Assert.That(expected4.ExpectedWeights.AsSpan().SequenceEqual([
        //  0  1  2  3  4  5  6  7  8  9 10
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        ])).IsTrue();
    }

    private sealed class TestTokenizer : ITokenizer<string>
    {
        private const string Tokens = "0123456789";
        public int TokenCount { get; }

        public IEnumerable<int> Tokenize(string data) => data.Select(c => Tokens.IndexOf(c));

        public int TokenizeSingle(string data) => data.Length is 1 ? Tokens.IndexOf(data) : throw new InvalidOperationException();


        public string GetToken(int data) => Tokens[data].ToString();

        public string Decode(IEnumerable<int> tokens) => string.Join("", tokens.Select(GetToken));

    }
}
