﻿using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Embedding;

public interface IEmbedder<in TInput, TOutput> : IEmbeddingLayer<TInput>, IUnembeddingLayer<TOutput>
{
    public Vector Embed(TInput input);
    public (TOutput output, Weight confidence) Unembed(Vector input);

    int IEmbeddingLayer<TInput>.OutputNodeCount => 0;
    int IUnembeddingLayer<TOutput>.InputNodeCount => 0;
    uint ILayer.ParameterCount => 0;

    Vector IEmbeddingLayer<TInput>.Forward(TInput input) => Embed(input);
    Vector IEmbeddingLayer<TInput>.Forward(TInput input, ILayerSnapshot snapshot) => throw new NotImplementedException();

    (TOutput output, Weight confidence) IUnembeddingLayer<TOutput>.Forward(Vector input) => Unembed(input);
    (TOutput output, int index, Vector weights) IUnembeddingLayer<TOutput>.Forward(Vector input, ILayerSnapshot snapshot) => throw new NotImplementedException();
}
