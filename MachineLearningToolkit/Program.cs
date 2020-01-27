using System;
using System.Collections.Generic;
using System.IO;

namespace MachineLearningToolkit
{
    public class Program
    {
        private static string graphFile = null;
        private static string labelFile = null;
        private static string listFile = "";
        private static string modelDir = "";
        private static string outputDir = "";
        private static string trainDir = "";
        private static string trainImagesDir = "";
        private static int trainingSteps;

        static void Main(string[] args)
        {
            try
            {
                if (args.Length == 0)
                {
                    Console.WriteLine("Informe os seguintes argumentos:\n" +
                        "--modelDir [Path absoluto ate a pasta que contem o grafo e o label map]\n" +
                        "--outputDir [Path de uma pasta em que serao armazenados os resultados temporariamente]\n" +
                        "--listFile [Path para o arquivo serializado com a lista de imagens e quadrantes.");
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
                        case "--trainDir":
                            trainDir = args[i + 1];
                            break;
                        case "--trainImagesDir":
                            trainImagesDir = args[i + 1];
                            break;
                        case "--trainingSteps":
                            trainingSteps = int.Parse(args[i + 1]);
                            break;
                    }
                }

                if (string.IsNullOrEmpty(modelDir) || string.IsNullOrEmpty(outputDir) || string.IsNullOrEmpty(listFile))
                    Console.WriteLine("Informe os parametros --modelDir, --listFile --outputDir");

                if (args[0] == "ObjectDetection")
                {
                    ObjectDetection test;

                    if (string.IsNullOrEmpty(graphFile) && string.IsNullOrEmpty(labelFile))
                    {
                        test = new ObjectDetection(modelDir);
                    }
                    else
                    {
                        test = new ObjectDetection(modelDir, graphFile, labelFile);
                    }

                    var results = test.Inference(listFile);

                    string outputFile = Path.Combine(outputDir, "Result.ObjectDetection");

                    JsonUtil<List<Result>>.WriteJsonOnFile(results, outputFile);

                    Console.WriteLine(outputFile);
                }
                else if (args[0] == "ImageClassification")
                {
                    ImageClassification test;

                    if (string.IsNullOrEmpty(graphFile) && string.IsNullOrEmpty(labelFile))
                    {
                        test = new ImageClassification(modelDir);
                    }
                    else
                    {
                        test = new ImageClassification(modelDir, graphFile, labelFile);
                    }

                    var results = test.Classify(listFile);

                    string outputFile = Path.Combine(outputDir, "Result.ImageClassification");

                    JsonUtil<List<ClassificationInference>>.WriteJsonOnFile(results, outputFile);

                    Console.WriteLine(outputFile);
                }
                else if (args[0] == "ImageClassificationRetrainer")
                {

                }
            }

            catch (IndexOutOfRangeException)
            {
                Console.WriteLine("Verifique o comando executado");
            }
            catch (Exception)
            {
                throw;
            }
        }
    }
}