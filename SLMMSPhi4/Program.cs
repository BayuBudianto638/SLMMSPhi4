using LLama.Common;
using LLama;
using LLama.Sampling;

string modelPath = @"E:\QQUF\Phi-3-mini-4k-instruct-q4.gguf";

var parameters = new ModelParams(modelPath)
{
    ContextSize = 1024,
    GpuLayerCount = 5
};
using var model = LLamaWeights.LoadFromFile(parameters);
using var context = model.CreateContext(parameters);
var executor = new InteractiveExecutor(context);

var chatHistory = new ChatHistory();
chatHistory.AddMessage(AuthorRole.System, "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.");
chatHistory.AddMessage(AuthorRole.User, "Hello, Bob.");
chatHistory.AddMessage(AuthorRole.Assistant, "Hello. How may I help you today?");

ChatSession session = new(executor, chatHistory);

InferenceParams inferenceParams = new InferenceParams()
{
    MaxTokens = 256,
    AntiPrompts = new List<string> { "User:" },

    SamplingPipeline = new DefaultSamplingPipeline(),
};

Console.ForegroundColor = ConsoleColor.Yellow;
Console.Write("The chat session has started.\nUser: ");
Console.ForegroundColor = ConsoleColor.Green;
string userInput = Console.ReadLine() ?? "";

while (userInput != "exit")
{
    await foreach (
        var text
        in session.ChatAsync(
            new ChatHistory.Message(AuthorRole.User, userInput),
            inferenceParams))
    {
        Console.ForegroundColor = ConsoleColor.White;
        Console.Write(text);
    }
    Console.ForegroundColor = ConsoleColor.Green;
    userInput = Console.ReadLine() ?? "";
}