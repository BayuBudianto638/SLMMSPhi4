using LLama.Common;
using LLama;
using LLama.Sampling;

string modelPath = @"E:\MiniLM\QQUF\Phi-3-mini-4k-instruct-q4.gguf";
string inputFilePath = @"E:\MiniLM\ocr_text.txt";  

var parameters = new ModelParams(modelPath)
{
    ContextSize = 1024,
    GpuLayerCount = 5
};

using var model = LLamaWeights.LoadFromFile(parameters);
using var context = model.CreateContext(parameters);
var executor = new InteractiveExecutor(context);

var chatHistory = new ChatHistory();
chatHistory.AddMessage(AuthorRole.System,
"You are a document cleaner. Given dirty OCR text, you only return the cleaned version as raw JSON. Do not add any extra words, labels, explanation, or prefixes like 'Assistant:'.");

ChatSession session = new(executor, chatHistory);

InferenceParams inferenceParams = new InferenceParams
{
    MaxTokens = 512,
    AntiPrompts = new List<string> { "User:" },
    SamplingPipeline = new DefaultSamplingPipeline()
};

if (!File.Exists(inputFilePath))
{
    Console.WriteLine("❌ File not found: " + inputFilePath);
    return;
}

string rawOcrText = await File.ReadAllTextAsync(inputFilePath);

string prompt = $@"Fix and complete the following OCR output and return JSON only. No need of explanation or extra words. Just return the cleaned text in JSON format.

OCR: {rawOcrText}

JSON Output Format:
{{""cleaned_text"": ""<corrected version>""}}
";



Console.WriteLine("\n--- LLaMA Output ---\n");
await foreach (var text in session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), inferenceParams))
{
    Console.ForegroundColor = ConsoleColor.White;
    Console.Write(text);
}
Console.ResetColor();
