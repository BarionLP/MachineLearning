namespace MachineLearning.Samples.Language;

public static class LMHelper
{
    private static readonly HashSet<char> EndSymbols = ['\0'];
    public static void StartChat(EmbeddedModel<int[], int> model, int contextSize, string tokens)
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
            Generate([.. input.Select(c => tokens.IndexOf(char.ToLower(c)))], model, contextSize, tokens);
        } while (true);
    }

    public static void Generate(int[] input, EmbeddedModel<int[], int> model, int contextSize, string tokens)
    {
        if (input.Contains(-1))
        {
            Console.WriteLine("Invalid Tokens detected");
            return;
        }

        int prediction;
        char token;
        Weight confidence;
        do
        {
            (prediction, confidence) = model.Process(input);
            token = tokens[prediction];
            input = [.. input, prediction];
            SetConsoleTextColor(confidence);
            Console.Write(token);
        } while (!EndSymbols.Contains(token) && input.Length < contextSize);
        Console.Write("\u001b[0m"); // reset color
        Console.WriteLine();

        static void SetConsoleTextColor(double confidence)
        {
            Console.Write($"\u001b[38;2;{(1 - confidence) * 255:F0};{confidence * 255:F0};60m");
        }
    }
}
