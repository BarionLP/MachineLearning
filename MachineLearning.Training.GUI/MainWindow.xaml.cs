using MachineLearning.Mamba;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;

namespace MachineLearning.Training.GUI;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public TrainingProgressTracker ProgressTracker { get; } = new();

    public ObservableCollection<TabItem> LayerViews { get; } = [];

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;

        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

        var model = new Mamba2VectorModel(6, 51, 64, 128, 32);
        model.Initialize();


        foreach(var (i, layer) in model.HiddenLayers.Index())
        {
            LayerViews.Add(new TabItem { Header = $"Layer {i}", Content = new LayerView([layer.C]) });
        }


        LayerViews.Consume(v => (v.Content as LayerView)?.Update());
    }
}