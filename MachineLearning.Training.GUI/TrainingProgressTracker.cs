using System.Collections.ObjectModel;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using LiveChartsCore.SkiaSharpView.Painting.Effects;
using MachineLearning.Data;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using SkiaSharp;

namespace MachineLearning.Training.GUI;

public sealed class TrainingProgressTracker
{
    public IEnumerable<ISeries<double>> EvaluationSeries => Entries.SelectMany<Entry, LineSeries<double>>(e => [e.Series, e.TrendSeries]);
    private readonly List<Entry> Entries = [];

    public EmbeddedModelTrainer<TInput, TOutput> CreateLinkedTrainer<TInput, TOutput>(string name, SKColor color, EmbeddedModel<TInput, TOutput> model, TrainingConfig config, ITrainingSet trainingset) where TInput : notnull where TOutput : notnull
    {
        var entry = new Entry(name, color);
        config = config with
        {
            EvaluationCallback = results => entry.Results.Add(results.Result.CorrectPercentage * 100),
        };
        var trainer = new EmbeddedModelTrainer<TInput, TOutput>(model, config, trainingset);
        Entries.Add(entry);
        return trainer;
    }

    public sealed class Entry
    {
        public LineSeries<double> Series { get; }
        public LineSeries<double> TrendSeries { get; }
        public readonly ObservableCollection<double> Results = [];
        public readonly ObservableCollection<double> Trends = [];

        public Entry(string name, SKColor color)
        {
            Series = new()
            {
                Values = Results,
                Name = name,
                Fill = null,
                GeometrySize = 5,
                GeometryStroke = new SolidColorPaint
                {
                    Color = color,
                    StrokeThickness = 3
                },
                Stroke = new SolidColorPaint
                {
                    StrokeThickness = 2,
                    Color = color,
                }
            };
            Results.CollectionChanged += (s, e) => UpdateTrends();
            TrendSeries = new()
            {
                Values = Trends,
                //Name = $"{name} - Trend",
                Fill = null,
                GeometrySize = 0,
                GeometryStroke = new SolidColorPaint
                {
                    Color = color,
                    StrokeThickness = 0
                },
                Stroke = new SolidColorPaint
                {
                    StrokeThickness = 2,
                    Color = color,
                    PathEffect = new DashEffect([4, 3]),
                },
            };
        }

        private void UpdateTrends()
        {
            Trends.Clear();
            if(Results.Count < 10)
                return;

            int windowSize = 7;
            for(int i = 0; i < Results.Count; i++)
            {
                var window = Results.Skip(Math.Max(0, i - windowSize + 1)).Take(Math.Min(windowSize, i + 1));
                Trends.Add(window.Average());
            }
        }

    }
}
