using System.Windows;
using MachineLearning.Data.Entry;
using MachineLearning.Data.Noise;
using MachineLearning.Data.Source;
using MachineLearning.Domain.Activation;
using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;
using SkiaSharp;

namespace MachineLearning.Training.GUI;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window {
    public TrainingProgressTracker ProgressTracker { get; } = new();
    

    public MainWindow() {
        InitializeComponent();
        DataContext = this;




        //var trainer = ProgressTracker.CreateLinkedTrainer("Text");

        Loaded += (sender, args) => 
        {
            //Task.Run(trainer.Train);
            //Task.Run(trainerTiny.Train);
        };
    }
}