using MachineLearningToolkit;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Tests
{
    [TestClass]
    public class ImageClassificationRetrainerTests
    {
        static string ImagesDir = @"C:\development\RetrainingImages";
        static string WorkspaceDir = @"C:\POD-VERA\Training";


        [TestMethod]
        public void Retraining()
        {
            Program.Main(new string[] { "ImageClassificationRetrainer", "--workspaceDir", WorkspaceDir,
                "--imagesDir", ImagesDir, "--logPath", "C:\\Logs\\MachineLearningToolkit.log" });
        }


    }
}
