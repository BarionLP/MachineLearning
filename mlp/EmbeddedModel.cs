using System.IO;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Serialization;

namespace ML.MultiLayerPerceptron;

public sealed class EmbeddedModel<TIn, TOut> : IEmbeddedModel<TIn, TOut>
{
    public required IEmbeddingLayer<TIn> InputLayer { get; init; }
    public required MultiLayerPerceptronModel InnerModel { get; init; }
    public required IUnembeddingLayer<TOut> OutputLayer { get; init; }

    public long WeightCount => InputLayer.WeightCount + InnerModel.WeightCount + OutputLayer.WeightCount;

    public (TOut prediction, Weight confidence) Process(TIn input)
        => OutputLayer.Process(InnerModel.Process(InputLayer.Process(input)));

    public override string ToString() => $"Embedded {InnerModel}";
}

public static class EmbeddedModel
{
    public static ErrorState Save(EmbeddedModel<int[], int> model, BinaryWriter writer)
    {
        if (model.InputLayer is not EncodedEmbeddingLayer eel)
        {
            return new NotImplementedException("EmbeddedModel<int[], int> only supports EncodedEmbeddingLayer rn");
        }

        var result = ModelSerializer.SaveEncodedEmbeddingLayer(eel, writer);
        if (!OptionsMarshall.IsSuccess(result))
        {
            return result;
        }

        result = MultiLayerPerceptronModel.Save(model.InnerModel, writer);
        if (!OptionsMarshall.IsSuccess(result))
        {
            return result;
        }

        if (model.OutputLayer is not TokenOutputLayer tol)
        {
            return new NotImplementedException("EmbeddedModel<int[], int> only supports TokenOutputLayer rn");
        }

        result = ModelSerializer.SaveTokenOutputLayer(tol, writer);
        if (!OptionsMarshall.IsSuccess(result))
        {
            return result;
        }

        return default;
    }

    public static Result<EmbeddedModel<int[], int>> Read(BinaryReader reader)
    {
        var input = ModelSerializer.ReadEncodedEmbeddingLayer(reader);
        if (!input.Branch(out _, out var error))
        {
            return error;
        }

        var inner = MultiLayerPerceptronModel.Read(reader);
        if (!inner.Branch(out _, out error))
        {
            return error;
        }

        var output = ModelSerializer.ReadTokenOutputLayer(reader);
        if (!output.Branch(out _, out error))
        {
            return error;
        }

        return new EmbeddedModel<int[], int>
        {
            InputLayer = input.OrThrow(),
            InnerModel = inner.OrThrow(),
            OutputLayer = output.OrThrow(),
        };
    }
}