using MachineLearningToolkit.Utility;
using NLog;
using NLog.Fluent;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Permissions;

namespace MachineLearningToolkit
{
    public class Program
    {
        private static readonly NLog.Logger Log = NLog.LogManager.GetCurrentClassLogger();

        private static string graphFile = null;
        private static string labelFile = null;
        private static string listFile = "";
        private static string logPath = "";
        private static string modelDir = "";
        private static string outputDir = "";
        private static string retrainerPath = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "ImageClassificationRetrainer.py");
        private static string trainDir = "";
        private static string trainImagesDir = "";
        private static int trainingSteps = 0;
        private static string tfhub_module_path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "InceptionV3.zip");
        internal static Process Process;
        public static void Main(string[] args)
        {
            try
            {
                if (args.Length == 0)
                {
                    Console.WriteLine("Informe a função que você deseja utilizar" +
                        "'ImageClassification', 'ObjectDetection' ou 'ClassificationRetrainer'");
                    //Console.WriteLine("Informe os seguintes argumentos:\n" +
                    //    "--modelDir [Path absoluto ate a pasta que contem o grafo e o label map]\n" +
                    //    "--outputDir [Path de uma pasta em que serao armazenados os resultados temporariamente]\n" +
                    //    "--listFile [Path para o arquivo serializado com a lista de imagens e quadrantes.");
                }

                for (int i = 0; i < args.Length; i++)
                {
                    string value = args[i];

                    switch (value)
                    {
                        case "--modelDir":
                            modelDir = PathNormalizer.NormalizeDirectory(args[i + 1]);
                            break;
                        case "--listFile":
                            listFile = PathNormalizer.NormalizeFilePath(args[i + 1]);
                            break;
                        case "--outputDir":
                            outputDir = PathNormalizer.NormalizeDirectory(args[i + 1]);
                            break;
                        case "--graphFile":
                            graphFile = PathNormalizer.NormalizeFilePath(args[i + 1]);
                            break;
                        case "--labelFile":
                            labelFile = PathNormalizer.NormalizeFilePath(args[i + 1]);
                            break;
                        case "--workspaceDir":
                            trainDir = PathNormalizer.NormalizeDirectory(args[i + 1]);
                            break;
                        case "--imagesDir":
                            trainImagesDir = PathNormalizer.NormalizeDirectory(args[i + 1]);
                            break;
                        case "--trainingSteps":
                            trainingSteps = int.Parse(args[i + 1]);
                            break;
                        case "--logPath":
                            if (args[i + 1] != "undefined")
                            {
                                logPath = PathNormalizer.NormalizeFilePath(args[i + 1]);
                            }
                            else
                            {
                                logPath = "";
                            }
                            break;
                    }
                }

                var config = new NLog.Config.LoggingConfiguration();

                // Targets where to log to: File and Console
                if (string.IsNullOrEmpty(logPath))
                {
                    logPath = "C:\\temp\\MachineLearningToolkit.log";
                }
                var logfile = new NLog.Targets.FileTarget("logfile") { FileName = logPath };

                // Rules for mapping loggers to targets            
                config.AddRule(LogLevel.Info, LogLevel.Fatal, logfile);

                // Apply config           
                NLog.LogManager.Configuration = config;
                if (args[0] != "ImageClassificationRetrainer")
                {
                    if (string.IsNullOrEmpty(modelDir) || string.IsNullOrEmpty(outputDir) || string.IsNullOrEmpty(listFile))
                        Log.Error("Informe os parametros --modelDir, --listFile --outputDir");
                }

                if (args[0] == "ObjectDetection")
                {
                    try
                    {
                        ObjectDetection inference;

                        Log.Info("Iniciando a detecção de objetos.");

                        if (string.IsNullOrEmpty(graphFile) && string.IsNullOrEmpty(labelFile))
                        {
                            inference = new ObjectDetection(modelDir);
                        }
                        else
                        {
                            inference = new ObjectDetection(modelDir, graphFile, labelFile);
                        }

                        var results = inference.Inference(listFile);

                        string outputFile = Path.Combine(outputDir, "Result.ObjectDetection");

                        JsonUtil<List<Result>>.WriteJsonOnFile(results, outputFile);

                        Log.Info("Detecção de objetos concluída.");

                        Console.WriteLine(outputFile);
                    }
                    catch (Exception ex)
                    {
                        Log.Error($"Houve um erro na detecção de objetos: {ex.Message}");
                    }
                }
                else if (args[0] == "ImageClassification")
                {
                    try
                    {
                        ImageClassification inference;

                        Log.Info("Iniciando classificação de imagens");

                        if (string.IsNullOrEmpty(graphFile) && string.IsNullOrEmpty(labelFile))
                        {
                            inference = new ImageClassification(modelDir);
                        }
                        else
                        {
                            inference = new ImageClassification(modelDir, graphFile, labelFile);
                        }

                        var results = inference.Classify(listFile);

                        string outputFile = Path.Combine(outputDir, "Result.ImageClassification");

                        JsonUtil<List<ClassificationInference>>.WriteJsonOnFile(results, outputFile);

                        Log.Info("Classificação de imagens concluída.");

                        Console.WriteLine(outputFile);
                    }
                    catch (Exception ex)
                    {
                        Log.Error($"Houve um erro na classificação de imagens: {ex.Message}");
                    }
                }
                else if (args[0] == "ImageClassificationRetrainer")
                {
                    try
                    {
                        string command = $"python {retrainerPath} --how_many_training_steps {trainingSteps} --image_dir {trainImagesDir} --destination_model_dir {outputDir} --log_path {logPath} --workspace_dir {trainDir} --tfhub_module_path {tfhub_module_path}";

                        var processInfo = new ProcessStartInfo("cmd.exe", "/c " + command);
                        processInfo.CreateNoWindow = true;
                        processInfo.UseShellExecute = false;
                        processInfo.RedirectStandardError = true;
                        processInfo.RedirectStandardOutput = true;

                        var process = Process.Start(processInfo);

                        process.OutputDataReceived += (object sender, DataReceivedEventArgs e) =>
                            Console.WriteLine("output>>" + e.Data);
                        process.BeginOutputReadLine();

                        process.ErrorDataReceived += (object sender, DataReceivedEventArgs e) =>
                            Console.WriteLine("error>>" + e.Data);
                        process.BeginErrorReadLine();

                        process.WaitForExit();

                        Console.WriteLine("ExitCode: {0}", process.ExitCode);
                        process.Close();

                        if (File.Exists($"{outputDir}\\retrained_graph.pb") && File.Exists($"{outputDir}\\label_map.txt"))
                        {
                            Console.WriteLine(outputDir);
                        }
                        else
                        {
                            string message = $"Houve um erro no processo de retreinamento do modelo de classificação de imagens: {trainDir}";
                            Log.Error(message);
                            throw new Exception(message);
                        }
                    }
                    catch (Exception ex)
                    {
                        if (ex.Message.Contains("Directory not empty"))
                        {
                            Log.Error($"Feche todas as aplicações ou visualizador de arquivos que estiverem utilizando o diretório de treinamento: {ex.Message}");
                        }

                        Directory.Delete(outputDir, true);
                        Directory.Delete(trainDir, true);
                    }
                    finally
                    {
                        Process?.Close();
                    }
                }
            }

            catch (IndexOutOfRangeException)
            {
                Log.Error("Verifique o comando executado");
            }
            catch (Exception ex)
            {
                Log.Error($"Erro interno: {ex.Message}");
            }
        }
    }
}