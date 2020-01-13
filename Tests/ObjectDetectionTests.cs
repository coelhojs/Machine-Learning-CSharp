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
        static string ModelDir = @"C:\Machine-Learning-Models-Server\models_inference\vera_poles_trees";
        static string OutputDir = "C:\\Machine-Learning-Models-Server\\test_images";
        static string OutputFile = Path.Combine(OutputDir, DateTime.Now.Ticks.ToString());

        [TestMethod]
        public string RequestJsonFileGenerator()
        {
            List<string> testList = new List<string>();

            for (int i = 0; i < 3; i++)
            {
                testList.Add("C:\\Machine-Learning-Models-Server\\test_images\\1.png");
            }
            //testList.Add("C:/Machine-Learning-Models-Server/test_images/2.png");
            //testList.Add("C:\\Machine-Learning-Models-Server\\test_images\\3.png");

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

            List<string> imagesList = RequestJsonFileReader(requestPath);

            ObjectDetection test = new ObjectDetection(ModelDir);

            var results = test.Inference(requestPath);

            string inferenceResultsFile = InferenceJsonFileGenerator(results);

            var parsedResults = InferenceJsonFileReader(inferenceResultsFile);

            Assert.IsNotNull(parsedResults);
        }


    }
}
