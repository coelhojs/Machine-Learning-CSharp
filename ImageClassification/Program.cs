using System;
using System.Collections.Generic;
using System.IO;
using MachineLearningToolkit.ImageClassification.Utility;

namespace MachineLearningToolkit.ImageClassification
{
    public class Program
    {
        private static string graphFile = null;
        private static string inputLayer = null;
        private static string labelFile = null;
        private static string listFile = "";
        private static string modelDir = "";
        private static string outputDir = "";
        private static string outputLayer = null;

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
                        case "--inputLayer":
                            inputLayer = args[i + 1];
                            break;
                        case "--outputLayer":
                            outputLayer = args[i + 1];
                            break;
                    }
                }

                if (string.IsNullOrEmpty(modelDir) || string.IsNullOrEmpty(outputDir) || string.IsNullOrEmpty(listFile))
                    Console.WriteLine("Informe os parametros --modelDir, --listFile --outputDir");

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

                string outputFile = Path.Combine(outputDir, DateTime.Now.Ticks.ToString());

                JsonUtil<List<Classification>>.WriteJsonOnFile(results, outputFile);

                Console.WriteLine(outputFile);
            }

            catch (IndexOutOfRangeException)
            {
                Console.WriteLine("Verifique o comando executado");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}