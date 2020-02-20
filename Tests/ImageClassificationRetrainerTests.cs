using MachineLearningToolkit;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Tests
{
    [TestClass]
    public class ImageClassificationRetrainerTests
    {
        static string ImagesDir = @"C:\development\RetrainingImages";
        static string WorkspaceDir = @"C:\POD-VERA\Training";
        static string OutputDir = @"C:\POD-VERA\MachineLearningModels\vera_species_2";


        [TestMethod]
        public void Retraining()
        {
            Program.Main(new string[] { "ImageClassificationRetrainer", "--workspaceDir", WorkspaceDir,
                "--trainingSteps", "1", "--outputDir", OutputDir, "--imagesDir", ImagesDir, "--log_path", "C:\\Logs\\MachineLearningToolkit.log" });
        }


    }
}
