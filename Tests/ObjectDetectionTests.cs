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
        static string ModelDir = @"C:\development\Vera\vera_base_trees\44668";
        //static string ModelDir = @"C:\development\Vera\vera_poles_trees_v1";
        static string OutputDir = "C:\\temp\\";
        static string OutputFile = Path.Combine(OutputDir, "Request.ObjectDetection");

        [TestMethod]
        public string RequestJsonFileGenerator()
        {
            List<string> testList = new List<string>();

            for (int i = 0; i < 3; i++)
            {
                testList.Add("C:\\temp\\arvore.jpg");
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

            Program.Main(new string[] { "ObjectDetection", "--modelDir", ModelDir,
                "--listFile", requestPath, "--outputDir", OutputDir, "--logPath", "C:\\Logs\\MachineLearningToolkit.log" });

            //ObjectDetection 
            //= new ObjectDetection(ModelDir);

            //var results = test.Inference(requestPath);

            //string inferenceResultsFile = InferenceJsonFileGenerator(results);

            //var parsedResults = InferenceJsonFileReader(inferenceResultsFile);

            //Assert.IsNotNull(parsedResults);
        }


    }
}
