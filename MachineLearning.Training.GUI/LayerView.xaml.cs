using SkiaSharp;
using System.Windows.Controls;

namespace MachineLearning.Training.GUI;

public partial class LayerView : UserControl
{
    private readonly Matrix[] weights;
    private readonly SKBitmap[] bitmaps;

    public LayerView(Matrix[] weights)
    {
        InitializeComponent();
        this.weights = weights;
        bitmaps = [.. weights.Select(m => new SKBitmap(m.ColumnCount, m.RowCount))];
    }

    public void Update()
    {
        GenerateHeatmap(weights[0],  bitmaps[0]);
    }

    private void canvas_PaintSurface(object sender, SkiaSharp.Views.Desktop.SKPaintSurfaceEventArgs e)
    {
        e.Surface.Canvas.Clear();

        float scaleX = (float) RenderSize.Width / bitmaps[0].Width;
        float scaleY = (float) RenderSize.Height / bitmaps[0].Height;
        float scale = float.Min(scaleX, scaleY);

        float targetWidth = bitmaps[0].Width * scale;
        float targetHeight = bitmaps[0].Height * scale;

        float left = ((float) RenderSize.Width - targetWidth) / 2;
        float top = ((float) RenderSize.Height - targetHeight) / 2;
        var destRect = new SKRect(left, top, left + targetWidth, top + targetHeight);

        e.Surface.Canvas.DrawBitmap(bitmaps[0], destRect);
    }

    public static SKBitmap GenerateHeatmap(Matrix matrix, SKBitmap bitmap)
    {
        var width = matrix.ColumnCount;
        var height = matrix.RowCount;

        var min = -0.5;
        var max = 0.5;
        var range = max - min;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double normalizedValue = (matrix[y, x] - min) / range;
                bitmap.SetPixel(x, y, normalizedValue < 0 ? SKColors.Black : normalizedValue > 1 ? SKColors.White : GetHeatmapColor(normalizedValue));
            }
        }

        return bitmap;
    }

    private static SKColor GetHeatmapColor(double value)
    {
        byte r = (byte) (255 * value);
        byte b = (byte) (255 * (1 - value));
        return new SKColor(r, 0, b, 255);
    }
}
