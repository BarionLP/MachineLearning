namespace MachineLearning.Samples.Language;

public static class LMHelper
{
    private const string EndSymbols = ".!?";
    public static void Generate(string input, IEmbeddedModel<string, char> model, int contextSize)
    {
        input = input.ToLowerInvariant();
        Console.Write(input);
        char prediction;
        Weight confidence;
        do
        {
            (prediction, confidence) = model.Forward(input);
            input += prediction;
            SetConsoleTextColor(confidence);
            Console.Write(prediction);
        } while (!EndSymbols.Contains(prediction) && input.Length < contextSize);
        Console.Write("\u001b[0m"); //reset color
        Console.WriteLine();

        static void SetConsoleTextColor(double confidence)
        {
            Console.Write($"\u001b[38;2;{(1 - confidence) * 255:F0};{confidence * 255:F0};60m");
        }
    }

    public static void StartChat(IEmbeddedModel<string, char> model, int contextSize)
    {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrEmpty(input))
            {
                return;
            }
            Generate(input, model, contextSize);
        } while (true);
    }
    public static void StartChat(IEmbeddedModel<int[], char> model, int contextSize, CharTokenizer tokenizer)
    {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrEmpty(input))
            {
                return;
            }
            Generate(tokenizer.Tokenize(input), model, contextSize);
        } while (true);
    }

    public static void Generate(int[] input, IEmbeddedModel<int[], char> model, int contextSize)
    {
        Console.Write(input);
        char prediction;
        Weight confidence;
        do
        {
            (prediction, confidence) = model.Forward(input);
            input = [.. input, prediction];
            SetConsoleTextColor(confidence);
            Console.Write(prediction);
        } while (!EndSymbols.Contains(prediction) && input.Length < contextSize);
        Console.Write("\u001b[0m"); //reset color
        Console.WriteLine();

        static void SetConsoleTextColor(double confidence)
        {
            Console.Write($"\u001b[38;2;{(1 - confidence) * 255:F0};{confidence * 255:F0};60m");
        }
    }
}
