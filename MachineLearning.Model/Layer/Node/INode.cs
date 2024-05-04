namespace MachineLearning.Model.Layer.Node;

public interface INode<TWeight>
{
    internal TWeight[] Weights { get; }
    internal TWeight Bias { get; set; }
}
