using MachineLearning.Model.Initialization;

namespace MachineLearning.Transformer;

public sealed class AttentionBlock(ModelInfo Info)
{
    public AttentionHead[] Heads { get; } = new AttentionHead[Info.AttentionHeadCountPerBlock];
    public ModelInfo Info { get; } = Info;

    public Matrix Process(Matrix input) {
        var result = Matrix.Create(input.RowCount, input.ColumnCount);
        foreach(var head in Heads) {
            //input.AddInPlace(head.GetEmbeddingDelta(input));
        }
        return result;
    }

    public int GetWeightsCount() => Heads[0].GetWeightsCount() * Info.AttentionHeadCountPerBlock;

    public void Initialize(IInitializer initializer) {
        Heads.ForEach(head => head.Initialize(initializer));
    }
}
