namespace Simple.Network.Layer.Node;

public interface INode<TWeight> {
    internal TWeight[] Weights { get; }
    internal TWeight Bias { get; set; }
}
