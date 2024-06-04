using System.Windows;
using MachineLearning.Samples;
using SkiaSharp;

namespace MachineLearning.Training.GUI;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public TrainingProgressTracker ProgressTracker { get; } = new();


    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;

        var model = BinaryClassifier.GetModel();
        var config = BinaryClassifier.GetTrainingConfig();


        var trainer = ProgressTracker.CreateLinkedTrainer("Binary Classifier", SKColors.Blue, config, model);

        Loaded += (sender, args) =>
        {
            Task.Run(trainer.Train);
            //Task.Run(trainerTiny.Train);
        };
    }
}