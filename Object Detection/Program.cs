namespace MachineLearningToolkit.ObjectDetection

{
    class Program
    {
        static void Main(string[] args)
        {
            string ModelDir = "C:\\Machine-Learning-Models-Server\\models_inference\\vera_poles_trees";
            string PicFile = "C:\\Machine-Learning-Models-Server\\test_images\\1.png";

            var detection = new ObjectDetection(ModelDir);
            InferenceResult result = detection.Inference(PicFile);
        }
    }
}
