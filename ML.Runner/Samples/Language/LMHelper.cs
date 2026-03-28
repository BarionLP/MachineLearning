using ML.Core.Data;
using ML.Core.Modules;

namespace ML.Runner.Samples.Language;

public static class LMHelper
{
    private static readonly HashSet<string> EndTokens = ["\0"];
    public static void StartChat(IEmbeddedModule<int[], int> model, int contextSize, ITokenizer<string> tokenizer)
    {
        var fillerToken = tokenizer.TokenizeSingle("\0");
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrEmpty(input))
            {
                return;
            }
            if (!Console.IsOutputRedirected)
            {
                Console.SetCursorPosition(0, Console.CursorTop - 1);
            }
            Console.Write(input);
            Generate([.. tokenizer.Tokenize(input)], model, contextSize, tokenizer, fillerToken);
        } while (true);
    }

    public static void Generate(int[] input, IEmbeddedModule<int[], int> model, int contextSize, ITokenizer<string> tokenizer, int fillerToken)
    {
        if (input.Contains(-1))
        {
            Console.WriteLine("Invalid Tokens detected");
            return;
        }

        int prediction;
        string token;
        Weight confidence;
        using var snapshot = model.CreateSnapshot();
        do
        {
            (prediction, confidence) = model.Forward(input, snapshot);
            token = tokenizer.GetToken(prediction);
            input = input[0] == fillerToken ? [.. input[1..], prediction] : [.. input, prediction];
            SetConsoleTextColor(confidence);
            Console.Write(token);
        } while (!EndTokens.Contains(token) && input.Length <= contextSize);
        Console.Write("End");
        Console.Write("\u001b[0m"); // reset color
        Console.WriteLine();

        static void SetConsoleTextColor(Weight confidence)
        {
            Console.Write($"\u001b[38;2;{(1 - confidence) * 255:F0};{confidence * 255:F0};60m");
        }
    }
}