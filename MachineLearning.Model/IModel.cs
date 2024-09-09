using System.Collections.Immutable;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public interface IModel<TLayer> where TLayer : ILayer
{
    public ImmutableArray<TLayer> Layers { get; }
    public TLayer OutputLayer { get; }

    public Vector Forward(Vector input);
}
