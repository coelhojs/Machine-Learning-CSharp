using System;
using System.Collections.Generic;
using System.IO;
using MachineLearningToolkit;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Tests
{
    [TestClass]
    public class ObjectDetectionTests
    {
        static string ModelDir = @"C:\POD-VERA\MachineLearningModels\vera_poles_trees";
        static string OutputDir = "C:\\development\\";
        static string OutputFile = Path.Combine(OutputDir, DateTime.Now.Ticks.ToString());

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

        public string InferenceJsonFileGenerator(List<Result> inferenceResults)
        {
            string resultsFile = Path.Combine(OutputDir, DateTime.Now.Ticks.ToString());

            JsonUtil<List<Result>>.WriteJsonOnFile(inferenceResults, resultsFile);

            Assert.IsNotNull(Path.GetFullPath(resultsFile));

            return resultsFile;
        }

        [TestMethod]
        public List<Result> InferenceJsonFileReader(string file)
        {
            List<Result> testList = JsonUtil<List<Result>>.ReadJsonFile(file);

            Assert.IsNotNull(testList);

            return testList;
        }

        [TestMethod]
        public void Inference()
        {
            string requestPath = RequestJsonFileGenerator();

            //List<string> imagesList = RequestJsonFileReader(requestPath);

            Program.Main(new string[] { "ObjectDetection", "--modelDir", @"C:\POD-VERA\MachineLearningModels\vera_poles_trees",
                "--listFile", requestPath, "--outputDir", "C:\\Temp", "--logPath", "C:\\Logs\\MachineLearningToolkit.log" });

            //ObjectDetection test = new ObjectDetection(ModelDir);

            //var results = test.Inference(requestPath);

            //string inferenceResultsFile = InferenceJsonFileGenerator(results);

            //var parsedResults = InferenceJsonFileReader(inferenceResultsFile);

            //Assert.IsNotNull(parsedResults);
        }


    }
}
