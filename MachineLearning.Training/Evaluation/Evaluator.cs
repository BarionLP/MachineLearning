using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using System.Net;

namespace MachineLearning.Training.Evaluation;

public static class Evaluator
{
    public static DataSetEvaluationResult Evaluate<TInput, TOutput, TLayer>(INetwork<TInput, TOutput, TLayer> model, IEnumerable<DataEntry<TInput, TOutput>> dataSet) where TInput : notnull where TOutput : notnull where TLayer : ILayer
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
}
