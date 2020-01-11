using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;
using MachineLearningToolkit.ObjectDetection.Utility;

namespace MachineLearningToolkit.ObjectDetection
{
    public class ObjectDetection
    {
        private const float MIN_SCORE = 0.7f;
        private Graph Graph;
        private NDArray ImgArr;
        private string LabelFile;
        private string ModelDir;
        private Session Session;
        public ObjectDetection(string modelDir, string graphFile = "frozen_inference_graph.pb", string labelFile = "label_map.pbtxt")
        {
            ModelDir = modelDir;
            Graph = ImportGraph(graphFile);
            LabelFile = labelFile;
            Session = tf.Session(Graph);
        }

        public InferenceResult Inference(string imagePath)
        {
            ImgArr = ReadTensorFromImageFile(Path.GetFullPath(imagePath));

            using (var sess = tf.Session(Graph))
                return Predict(sess, imagePath);


        }

        private Graph ImportGraph(string graphFile)
        {
            var graph = new Graph().as_default();
            graph.Import(Path.Join(ModelDir, graphFile));

            return graph;
        }

        private InferenceResult Predict(Session sess, string imagePath)
        {
            var graph = tf.get_default_graph();

            Tensor tensorNum = graph.OperationByName("num_detections");
            Tensor tensorBoxes = graph.OperationByName("detection_boxes");
            Tensor tensorScores = graph.OperationByName("detection_scores");
            Tensor tensorClasses = graph.OperationByName("detection_classes");
            Tensor imgTensor = graph.OperationByName("image_tensor");
            Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };

            var results = sess.run(outTensorArr, new FeedItem(imgTensor, ImgArr));

            return ParseInferenceResults(results, imagePath);
        }

        private NDArray ReadTensorFromImageFile(string file_name)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.read_file(file_name, "file_reader");
            var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            var casted = tf.cast(decodeJpeg, TF_DataType.TF_UINT8);
            var dims_expander = tf.expand_dims(casted, 0);

            using (var sess = tf.Session(graph))
                return sess.run(dims_expander);
        }

        private InferenceResult ParseInferenceResults(NDArray[] resultArr, string imagePath)
        {
            var detectionsList = new List<DetectionVO>();

            // get pbtxt items
            PbtxtItems pbTxtItems = PbtxtParser.ParsePbtxtFile(Path.Join(ModelDir, LabelFile));
            // get bitmap
            Bitmap bitmap = new Bitmap(Path.GetFullPath(imagePath));

            var detectionClasses = np.squeeze(resultArr[3]).GetData<float>();
            var detectionScores = resultArr[2].AsIterator<float>();
            var detectionBoxes = resultArr[1].GetData<float>().ToArray();

            var scores = detectionScores.Where(score => score > MIN_SCORE).ToArray();
            var classes = detectionClasses.Take(scores.Length).ToArray();

            for (int i = 0; i < scores.Length; i++)
            {
                detectionsList.Add(new DetectionVO()
                {
                    BoundingBox = CreateReactangle(bitmap, detectionBoxes, i),
                    Class = pbTxtItems.items.Where(w => w.id == detectionClasses[i]).Select(s => s.display_name).FirstOrDefault(),
                    ImagePath = imagePath,
                    Score = scores[i]
                });
            }

            return new InferenceResult()
            {
                NumDetections = scores.Length,
                Results = detectionsList
            };
        }

        private Rectangle CreateReactangle(Bitmap bitmap, float[] boxes, int i)
        {
            float top = boxes[i * 4] * bitmap.Height;
            float left = boxes[i * 4 + 1] * bitmap.Width;
            float bottom = boxes[i * 4 + 2] * bitmap.Height;
            float right = boxes[i * 4 + 3] * bitmap.Width;

            Rectangle rect = new Rectangle()
            {
                X = (int)left,
                Y = (int)top,
                Width = (int)(right - left),
                Height = (int)(bottom - top)
            };

            return rect;
        }
    }
}
