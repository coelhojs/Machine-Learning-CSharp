using MachineLearningToolkit.Utility;
using NLog;
using NLog.Fluent;
using System;
using System.Collections.Generic;
using System.IO;
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
        private static string trainDir = "";
        private static string trainImagesDir = "";
        private static int trainingSteps = 0;
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
                            modelDir = args[i + 1];
                            break;
                        case "--listFile":
                            listFile = args[i + 1];
                            break;
                        case "--outputDir":
                            outputDir = args[i + 1];
                            break;
                        case "--graphFile":
                            graphFile = args[i + 1];
                            break;
                        case "--labelFile":
                            labelFile = args[i + 1];
                            break;
                        case "--workspaceDir":
                            trainDir = args[i + 1];
                            break;
                        case "--imagesDir":
                            trainImagesDir = args[i + 1];
                            break;
                        case "--trainingSteps":
                            trainingSteps = int.Parse(args[i + 1]);
                            break;
                        case "--logPath":
                            logPath = args[i + 1];
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

                if (string.IsNullOrEmpty(modelDir) || string.IsNullOrEmpty(outputDir) || string.IsNullOrEmpty(listFile))
                    Log.Error("Informe os parametros --modelDir, --listFile --outputDir");

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
                        ImageClassificationRetrainer retrainer;

                        Log.Info("Iniciando retreinamento do modelo de classificação de imagens");

                        if (string.IsNullOrEmpty(trainImagesDir) && trainingSteps == 0)
                        {
                            retrainer = new ImageClassificationRetrainer(trainDir);
                        }
                        else if (trainingSteps == 0)
                        {
                            retrainer = new ImageClassificationRetrainer(trainDir, trainImagesDir);
                        }
                        else if (string.IsNullOrEmpty(trainImagesDir))
                        {
                            retrainer = new ImageClassificationRetrainer(trainDir, "", trainingSteps);
                        }
                        else
                        {
                            retrainer = new ImageClassificationRetrainer(trainDir, trainImagesDir);
                        }

                        var trainingResults = retrainer.Retrain();

                        if (trainingResults)
                        {
                            //string outputFile = Path.Combine(trainDir, "TrainingResults.ImageClassificationRetrainer");
                            Log.Info("Treinamento concluído com acurácia superior a 75%.");
                            //Console.WriteLine(outputFile);
                        }
                    }
                    catch (Exception ex)
                    {
                        Log.Error($"Houve um erro no retreinamento do modelo de classificação de imagens: {ex.Message}");
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