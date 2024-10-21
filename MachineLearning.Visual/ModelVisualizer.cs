using MachineLearning.Model;
using MachineLearning.Model.Layer;
using System.Drawing;
using System.Drawing.Imaging;

namespace MachineLearning.Visual;

public static class ModelVisualizer
{
    public static void Visualize(IEmbeddedModel<string, char> model, DirectoryInfo path)
    {
        path.CreateIfNotExists();
        var count = 0;

        foreach(var layer in model.Layers)
        {
            count++;
            var map = layer switch
            {
                SimpleLayer sl => GenerateHeatmap(sl.Weights),
                StringEmbeddingLayer el => GenerateHeatmap(el.EmbeddingMatrix),
                _ => null,
            };

            map?.Save(path.File($"layer_{count}.png").FullName, ImageFormat.Png);
        }
    }

    public static Bitmap GenerateHeatmap(Matrix matrix)
    {
        var width = matrix.ColumnCount;
        var height = matrix.RowCount;
        var bitmap = new Bitmap(width, height);

        var min = -0.5;
        var max = 0.5;
        var range = max - min;

        for(int y = 0; y < height; y++)
        {
            for(int x = 0; x < width; x++)
            {
                double normalizedValue = (matrix[y, x] - min) / range;
                bitmap.SetPixel(x, y, normalizedValue < 0 ? Color.Black : normalizedValue > 1 ? Color.White : GetHeatmapColor(normalizedValue));
            }
        }

        return bitmap;
    }

    private static Color GetHeatmapColor(double value)
    {
        int r = (int) (255 * value);
        int b = (int) (255 * (1 - value));
        return Color.FromArgb(255, r, 0, b);
    }
}

public static class ModelAnalyzer
{
    public static void Analyze(IEnumerable<ILayer> layers)
    {
        foreach(var layer in layers)
        {
            switch(layer)
            {
                case StringEmbeddingLayer em:
                    Console.WriteLine($"Embedding Layer: Av: {em.EmbeddingMatrix.Sum() / em.EmbeddingMatrix.FlatCount:F4}; Max: {em.EmbeddingMatrix.Max():F4}; Min: {em.EmbeddingMatrix.Min():F4}");
                    break;

                case SimpleLayer sl:
                    Console.WriteLine($"Simple Layer:");
                    Console.WriteLine($"\tWeights: Av: {sl.Weights.Sum()/sl.Weights.FlatCount:F4}; Max: {sl.Weights.Max():F4}; Min: {sl.Weights.Min():F4}");
                    Console.WriteLine($"\tBiases: Av: {sl.Biases.Sum()/sl.Biases.Count:F4}; Max: {sl.Biases.Max():F4}; Min: {sl.Biases.Min():F4}");
                    break;
            }
        }
    }
}
