using System;
using System.Collections.Generic;
using System.IO;
using MachineLearningToolkit;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Tests
{
    [TestClass]
    public class ImageClassificationTests
    {
        static string ModelDir = @"C:\POD-VERA\MachineLearningModels\vera_species";
        static string OutputDir = "C:\\temp";
        static string OutputFile = Path.Combine(OutputDir, "Request.ImageClassification");

        [TestMethod]
        public string RequestJsonFileGenerator()
        {
            List<string> testList = new List<string>();

            for (int i = 0; i < 3; i++)
            {
                testList.Add("C:\\development\\1.png");
            }

            JsonUtil<List<string>>.WriteJsonOnFile(testList, OutputFile);

            Assert.IsNotNull(Path.GetFullPath(OutputFile));

            return OutputFile;
        }

        [TestMethod]
        public List<string> RequestJsonFileReader(string file)
        {
            List<string> testList = JsonUtil<List<string>>.ReadJsonFile(file);

            Assert.IsNotNull(testList);

            return testList;
        }

        public string InferenceJsonFileGenerator(List<ClassificationInference> inferenceResults)
        {
            string resultsFile = Path.Combine(OutputDir, DateTime.Now.Ticks.ToString());

            JsonUtil<List<ClassificationInference>>.WriteJsonOnFile(inferenceResults, resultsFile);

            Assert.IsNotNull(Path.GetFullPath(resultsFile));

            return resultsFile;
        }

        [TestMethod]
        public List<ClassificationInference> InferenceJsonFileReader(string file)
        {
            List<ClassificationInference> testList = JsonUtil<List<ClassificationInference>>.ReadJsonFile(file);

            Assert.IsNotNull(testList);

            return testList;
        }

        [TestMethod]
        public void Classification()
        {
            string requestPath = RequestJsonFileGenerator();

            Program.Main(new string[] { "ImageClassification", "--modelDir", @"C:\POD-VERA\MachineLearningModels\vera_species",
                "--listFile", requestPath, "--outputDir", "C:\\temp", "--logPath", "C:\\Logs\\MachineLearningToolkit.log" });

            //List<string> imagesList = RequestJsonFileReader(requestPath);

            //ImageClassification test = new ImageClassification(ModelDir);

            //var results = test.Classify(requestPath);

            //string inferenceResultsFile = InferenceJsonFileGenerator(results);

            //var parsedResults = InferenceJsonFileReader(inferenceResultsFile);

            //Assert.IsNotNull(parsedResults);
        }
    }
}