﻿using MachineLearning.Data.Noise;
using MachineLearning.Data.Source;
using ModelDefinition = MachineLearning.Model.SimpleNetwork<double[], int, MachineLearning.Model.Layer.RecordingLayer>;

namespace MachineLearning.Samples.MNIST;

public class MNISTModel
{
    public static ModelDefinition GetModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);

        var network = ModelBuilder.Recorded<double[], int>(784)
                .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
                .SetEmbedder(MNISTEmbedder.Instance)
                .AddLayer(256, initializer)
                .AddLayer(128, initializer)
                .AddLayer(10, builder => builder.SetActivationMethod(SoftmaxActivation.Instance).Initialize(new XavierInitializer(random)))
                .Build();

        return network;
    }

    public static TrainingConfig<double[], int> GetTrainingConfig(Random? random = null)
    {
        var dataSource = new MNISTDataSource(AssetManager.MNISTArchive);

        return new TrainingConfig<double[], int>()
        {
            TrainingSet = dataSource.TrainingSet,
            TestSet = dataSource.TestingSet,

            EpochCount = 8,
            BatchCount = 128,

            Optimizer = new AdamOptimizerConfig
            {
                LearningRate = 0.1,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new MNISTOutputResolver(),
            InputNoise = new ImageInputNoise
            {
                Size = ImageDataEntry.SIZE,
                NoiseStrength = 0.35,
                NoiseProbability = 0.75,
                MaxShift = 2,
                MaxAngle = 30,
                MinScale = 0.8,
                MaxScale = 1.2,
                Random = random ?? Random.Shared,
            },

            DumpEvaluationAfterBatches = 8,
            EvaluationCallback = evaluation => Console.WriteLine(evaluation.Dump()),

            ShuffleTrainingSetPerEpoch = true,

            RandomSource = random ?? Random.Shared,
        };
    }

    public static ModelDefinition TrainDefault(Random? random = null)
    {
        var model = GetModel(random);
        var config = GetTrainingConfig(random);

        TrainDefault(model, config);

        return model;
    }

    public static void TrainDefault(ModelDefinition model, TrainingConfig<double[], int> config)
    {
        var trainer = ModelTrainer.Create(model, config);

        trainer.Train();

        var images = new ImageDataSource(AssetManager.CustomDigits);
        var correctCounter = 0;
        var counter = 0;
        var previousColor = Console.ForegroundColor;
        foreach(var image in images.DataSet)
        {
            var prediction = model.Process(image.Image);

            if(prediction == image.Digit)
            {
                correctCounter++;
            }

            Console.ForegroundColor = prediction == image.Digit ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"Predicted: {prediction}\tActual: {image.Digit}");
            counter++;
        }
        Console.ForegroundColor = previousColor;
        Console.WriteLine($"Correct: {(double) correctCounter / counter:P0}");
    }
}
