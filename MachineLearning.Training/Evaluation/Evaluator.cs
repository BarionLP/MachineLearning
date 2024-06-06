using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;
using System.Net;

namespace MachineLearning.Training.Evaluation;

public static class Evaluator
{
    public static DataSetEvaluationResult Evaluate<TInput, TOutput>(
        this EmbeddedModel<TInput, TOutput> model, 
        IEnumerable<DataEntry<TInput, TOutput>> dataSet
        ) where TInput : notnull where TOutput : notnull
    {
        int correctCounter = 0;
        int totalCounter = 0;
        foreach(var entry in dataSet)
        {
            totalCounter++;
            if(model.Process(entry.Input).Equals(entry.Expected))
            {
                correctCounter++;
            }
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = double.NaN,
        };
    }

    public static DataSetEvaluationResult Evaluate<TInput, TOutput>(
        this EmbeddedModel<TInput, TOutput> model, 
        ICostFunction costFunction, 
        IOutputResolver<TOutput> outputResolver, 
        IEnumerable<DataEntry<TInput, TOutput>> dataSet
        ) where TInput : notnull where TOutput : notnull
    {
        int correctCounter = 0;
        double totalCost = 0;
        int totalCounter = 0;
        foreach(var entry in dataSet)
        {
            totalCounter++;
            var outputWeights = model.InternalModel.Forward(model.Embedder.Embed(entry.Input));
            var output = model.Embedder.UnEmbed(outputWeights);

            if(output.Equals(entry.Expected))
            {
                correctCounter++;
            }

            totalCost += costFunction.TotalCost(outputWeights, outputResolver.Expected(entry.Expected));
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }
}
