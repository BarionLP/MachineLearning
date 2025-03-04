﻿using System.Globalization;
using System.Windows;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Samples;
using MachineLearning.Samples.MNIST;
using MachineLearning.Serialization;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization.Adam;
using MachineLearning.Training.Optimization.Nadam;
using MachineLearning.Training.Optimization.SGDMomentum;
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

        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

        //var serializer = new ModelSerializer(AssetManager.GetModelFile("mnist.nnw"));
        var model1 = MNISTModel.CreateModel(new Random(42));
        var model2 = MNISTModel.CreateModel(new Random(42));
        //var model = serializer.Load(MNISTEmbedder.Instance).ReduceOrThrow();
        var config = MNISTModel.DefaultTrainingConfig();

        // var trainer1 = ProgressTracker.CreateLinkedTrainer("Adam Optimizer", SKColors.Blue, model1, config with { Optimizer = new AdamOptimizer
        // {
        //     LearningRate = 0.1f,
        //     CostFunction = CrossEntropyLoss.Instance,
        // },
        // DumpEvaluationAfterBatches = 4,
        // });
        
        // var trainer2 = ProgressTracker.CreateLinkedTrainer("Nadam Optimizer", SKColors.Red, model2, config with { Optimizer = new NadamOptimizer
        // {
        //     LearningRate = 0.1f,
        //     CostFunction = CrossEntropyLoss.Instance,
        // },
        // DumpEvaluationAfterBatches = 4,

        // });

        Loaded += (sender, args) =>
        {
            // StatusLabel.Content = "Training...";
            // _ = Task.Run(() => trainer1.Train());
            // _ = Task.Run(() => trainer2.Train());
            //serializer.Save(model);
            //StatusLabel.Content = "Done!";
            //Task.Run(trainerTiny.Train);
        };
    }
}