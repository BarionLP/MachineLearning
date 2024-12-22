using MachineLearning.Data;

namespace MachineLearning.Samples.Language;

public static class LMHelper
{
    private static readonly HashSet<string> EndTokens = ["\0"];
    public static void StartChat(EmbeddedModel<int[], int> model, int contextSize, ITokenizer<string> tokenizer)
    {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrEmpty(input))
            {
                return;
            }
            Console.Write(input);
            Generate([.. tokenizer.Tokenize(input)], model, contextSize, tokenizer);
        } while (true);
    }

    public static void Generate(int[] input, EmbeddedModel<int[], int> model, int contextSize, ITokenizer<string> tokenizer)
    {
        if (input.Contains(-1))
        {
            Console.WriteLine("Invalid Tokens detected");
            return;
        }

        int prediction;
        string token;
        Weight confidence;
        do
        {
            (prediction, confidence) = model.Process(input);
            token = tokenizer.GetToken(prediction);
            input = [.. input, prediction];
            SetConsoleTextColor(confidence);
            Console.Write(token);
        } while (!EndTokens.Contains(token) && input.Length < contextSize);
        Console.Write("\u001b[0m"); // reset color
        Console.WriteLine();

        static void SetConsoleTextColor(double confidence)
        {
            Console.Write($"\u001b[38;2;{(1 - confidence) * 255:F0};{confidence * 255:F0};60m");
        }
    }
}
