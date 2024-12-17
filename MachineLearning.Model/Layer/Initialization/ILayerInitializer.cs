namespace MachineLearning.Model.Layer.Initialization;

public interface IInitializer<TValue> 
{
    public void Initialize(TValue layer);
}

public sealed class NoInitializer<T> : IInitializer<T>
{
    public static NoInitializer<T> Instance { get; } = new();
    public void Initialize(T layer) { }
} 