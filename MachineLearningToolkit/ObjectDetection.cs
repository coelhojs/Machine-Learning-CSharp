using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using MachineLearningToolkit.Utility;
using NLog;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MachineLearningToolkit
{
    public class ObjectDetection
    {
        private static readonly Logger Log = LogManager.GetCurrentClassLogger();

        private Graph Graph;
        private PbtxtItems Labels;
        private int MaxDetections;
        private float MinScore;
        private string ModelDir;
        private string OutputDir;

        public ObjectDetection(string modelDir, int maxDetections, float minScore, string outputDir, string graphFile = "frozen_inference_graph.pb", string labelFile = "label_map.pbtxt")
        {
            ModelDir = modelDir;
            Graph = ImportGraph(graphFile);
            Labels = LoadLabels(labelFile);
            MaxDetections = maxDetections;
            MinScore = minScore;
            OutputDir = outputDir;
        }
        public List<Result> Inference(string listPath, bool drawImages = false)
        {
            try
            {
                var Results = new List<Result>();

                var list = JsonUtil<List<string>>.ReadJsonFile(listPath);

                foreach (var image in list)
                {
                    Log.Info($"Lendo imagem {Path.GetFileName(image)}");

                    NDArray imgArr = ReadTensorFromImageFile(Security.GrantAccess(Path.GetFullPath(image)));

                    using (var sess = tf.Session(Graph))
                        Results.Add(Predict(sess, imgArr, image, drawImages));
                }
                return Results;
            }
            catch (Exception ex)
            {
                Log.Error($"Não foi possível executar a detecção para o grupo de imagens atual: {ex.Message}");
                throw ex;
            }
        }
        private Graph ImportGraph(string graphFile)
        {
            try
            {
                var graph = new Graph().as_default();
                graph.Import(Security.GrantAccess(Path.Combine(ModelDir, graphFile)));

                Log.Info($"Modelo carregado do diretório {ModelDir} com sucesso.");

                return graph;
            }
            catch (Exception ex)
            {
                Log.Error("Não foi possivel carregar o arquivo do modelo.\nVerifique o path para o arquivo .pb " +
                    $"com o argumento --graphFile: {ex.Message}");
                throw ex;
            }
        }
        private Result Predict(Session sess, NDArray imgArr, string image, bool drawImages)
        {
            try
            {
                var graph = tf.get_default_graph();

                Tensor tensorNum = graph.OperationByName("num_detections");
                Tensor tensorBoxes = graph.OperationByName("detection_boxes");
                Tensor tensorScores = graph.OperationByName("detection_scores");
                Tensor tensorClasses = graph.OperationByName("detection_classes");
                Tensor imgTensor = graph.OperationByName("image_tensor");
                Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };

                Log.Info($"Executando detecção");

                var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));

                return ParseResults(results, image, drawImages);
            }
            catch (FileNotFoundException ex)
            {
                Log.Error($"Arquivo {ex.Message} não encontrado.");

                return new Result()
                {
                    Error = new KeyValuePair<string, string>("FileNotFound", image)
                };
            }
            catch (Exception ex)
            {
                Log.Error(ex.Message);
                throw;
            }
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

        private Result ParseResults(NDArray[] resultArr, string imagePath, bool drawImages)
        {
            var detectionsList = new List<DetectionInference>();

            using (Bitmap bitmap = new Bitmap(Security.GrantAccess(Path.GetFullPath(imagePath))))
            {
                var detectionClasses = np.squeeze(resultArr[3]).GetData<float>();
                var detectionScores = resultArr[2].AsIterator<float>();
                var detectionBoxes = resultArr[1].GetData<float>().ToArray();

                var scores = detectionScores.Where(score => score > MinScore).ToArray();
                var classes = detectionClasses.Take(scores.Length).ToArray();

                if (MaxDetections == 1 && scores.Length > 0)
                {
                    int index = FindHighestScore(scores);

                    var detection = new DetectionInference()
                    {
                        BoundingBox = CreateReactangle(bitmap, detectionBoxes, index),
                        Class = Labels.items.Where(w => w.id == detectionClasses[index]).Select(s => s.display_name).FirstOrDefault(),
                        ImagePath = imagePath,
                        Score = scores[index],
                        DateTime = DateTime.Now
                    };

                    DrawObjectOnBitmap(bitmap, detection.BoundingBox, detection.Score, detection.Class);

                    detectionsList.Add(detection);

                    Log.Info($"{detection.Class} detectado(a) na imagem {Path.GetFileName(detection.ImagePath)} com probabilidade de {detection.Score}");
                }
                else
                {
                    for (int i = 0; i < scores.Length; i++)
                    {
                        var detection = new DetectionInference()
                        {
                            BoundingBox = CreateReactangle(bitmap, detectionBoxes, i),
                            Class = Labels.items.Where(w => w.id == detectionClasses[i]).Select(s => s.display_name).FirstOrDefault(),
                            ImagePath = imagePath,
                            Score = scores[i],
                            DateTime = DateTime.Now
                        };

                        DrawObjectOnBitmap(bitmap, detection.BoundingBox, detection.Score, detection.Class);

                        detectionsList.Add(detection);

                        Log.Info($"{detection.Class} detectado(a) na imagem {Path.GetFileName(detection.ImagePath)} com probabilidade de {detection.Score}");
                    }
                }

                if (drawImages)
                    bitmap.Save(Path.Combine(OutputDir, Path.GetFileName(imagePath)));

                Log.Info($"{scores.Length} objetos foram detectados na imagem {Path.GetFileName(imagePath)} e serão incluídos no arquivo de resposta");
            }

            return new Result()
            {
                DateTime = DateTime.Now,
                NumDetections = detectionsList.Count,
                Results = detectionsList
            };
        }

        private PbtxtItems LoadLabels(string labelFile)
        {
            try
            {
                // get pbtxt items
                var labels = PbtxtParser.ParsePbtxtFile(Security.GrantAccess(Path.Combine(ModelDir, labelFile)));

                Log.Info($"Labels do modelo carregadas: {String.Join(", ", labels.items)}");

                return labels;
            }
            catch (Exception ex)
            {
                Log.Error("Nao foi possivel carregar o arquivo de labels." +
                                   "\nVerifique o formato do arquivo e " +
                                   "informe o path para o arquivo .pbtxt " +
                                   $"com o argumento --labelFile: {ex.Message}");
                throw ex;
            }
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
        private int FindHighestScore(float[] scores)
        {
            float highestValue = 0.0f;
            int indexOf = 0;
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[i] > highestValue)
                {
                    highestValue = scores[i];
                    indexOf = i;
                }
            }
            return indexOf;
        }

        private void DrawObjectOnBitmap(Bitmap bmp, Rectangle rect, float score, string name)
        {
            Brush brush = Brushes.Red;
            Color color = Color.Red;

            switch (name)
            {
                case "arvore":
                    brush = Brushes.GreenYellow;
                    color = Color.GreenYellow;
                    break;
                case "poste":
                    brush = Brushes.Aqua;
                    color = Color.Aqua;
                    break;
                case "chave":
                    brush = Brushes.Teal;
                    color = Color.Teal;
                    break;
                case "baixa-tensao":
                    brush = Brushes.PeachPuff;
                    color = Color.PeachPuff;
                    break;
                case "media-tensao":
                    brush = Brushes.Salmon;
                    color = Color.Salmon;
                    break;
                case "transformador":
                    brush = Brushes.Blue;
                    color = Color.Blue;
                    break;
                default:
                    break;
            }

            using (Graphics graphic = Graphics.FromImage(bmp))
            {
                graphic.SmoothingMode = SmoothingMode.AntiAlias;

                using (Pen pen = new Pen(color, 8))
                {
                    graphic.DrawRectangle(pen, rect);

                    Point p = new Point(rect.Left + 5, rect.Bottom - 5);
                    string text = string.Format("{0}:{1}%", name, (int)(score * 100));
                    graphic.DrawString(text, new Font("Arial", 48), brush, p);
                }
            }
        }

        private static Image DrawBoundingBox(DetectionInference inference, Image image, Brush brush, Color color)
        {
            using (Graphics graphic = Graphics.FromImage(image))
            {
                graphic.SmoothingMode = SmoothingMode.AntiAlias;

                using (Pen pen = new Pen(color, 8))
                {
                    graphic.DrawRectangle(pen, inference.BoundingBox);

                    Point p = new Point(inference.BoundingBox.Left + 5, inference.BoundingBox.Bottom - 5);
                    string text = string.Format("{0}:{1}%", inference.Class, (int)(inference.Score * 100));
                    graphic.DrawString(text, new Font("Arial Black", 48), brush, p);
                }
            }
            return image;
        }

        private static Image WriteTextOnImage(string imagePath, string text)
        {
            Image myImage = Image.FromFile(imagePath, true);

            using (Graphics graphic = Graphics.FromImage(myImage))
            {
                graphic.SmoothingMode = SmoothingMode.AntiAlias;

                using (Pen pen = new Pen(Color.OrangeRed, 2))
                {
                    Point p = new Point(5, 5);

                    graphic.DrawString(text, new Font("Verdana", 96), Brushes.OrangeRed, p);
                }
            }

            return myImage;
        }

        /// <summary> 
        /// Cria um MemoryStream com a nova imagem com qualidade (dada em %) desejada
        /// </summary> 
        /// <param name="quality"> An integer from 0 to 100, with 100 being the highest quality. </param> 
        public static MemoryStream GenerateImageMS(Image img, int quality = 100)
        {

            if (quality < 0 || quality > 100)
            {
                throw new ArgumentOutOfRangeException("Qualidade deve estar entre 0 e 100.");
            }

            ImageCodecInfo jpegCodec = GetEncoderInfo("image/jpeg");

            EncoderParameter qualityParam = new EncoderParameter(Encoder.Quality, quality);
            EncoderParameter compressionParam = new EncoderParameter(Encoder.Compression, (long)EncoderValue.CompressionLZW);
            EncoderParameters encoderParams = new EncoderParameters(2);
            encoderParams.Param[0] = qualityParam;
            encoderParams.Param[1] = compressionParam;

            var ms = new MemoryStream();
            img.Save(ms, jpegCodec, encoderParams);

            return ms;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="mimeType"></param>
        /// <returns></returns>
        private static ImageCodecInfo GetEncoderInfo(string mimeType)
        {
            // Get image codecs for all image formats 
            ImageCodecInfo[] codecs = ImageCodecInfo.GetImageEncoders();

            // Find the correct image codec 
            for (int i = 0; i < codecs.Length; i++)
            {
                if (codecs[i].MimeType == mimeType)
                {
                    return codecs[i];
                }
            }
            return null;
        }
    }
}