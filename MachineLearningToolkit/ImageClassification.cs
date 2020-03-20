using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MachineLearningToolkit.Utility;
using NLog.Fluent;
using NumSharp;
using Tensorflow;
using static Tensorflow.Binding;

namespace MachineLearningToolkit
{
    public class ImageClassification
    {
        private static readonly NLog.Logger Log = NLog.LogManager.GetCurrentClassLogger();

        private Graph Graph;
        private string Input_name;
        private string[] Labels;
        private string ModelDir;
        private string Output_name;

        public ImageClassification(string modelDir,
                                    string graphFile = "retrained_graph.pb", string labelFile = "label_map.txt",
                                    string input_name = "Placeholder", string output_name = "final_result")
        {
            ModelDir = modelDir;
            Graph = ImportGraph(graphFile);
            Input_name = input_name;
            Labels = LoadLabels(modelDir, labelFile);
            Output_name = output_name;
        }
        private Graph ImportGraph(string graphFile)
        {
            try
            {
                var graph = new Graph();
                graph.Import(Security.GrantAccess(Path.Combine(ModelDir, graphFile)));

                Log.Info($"Modelo carregado do diretório {ModelDir} com sucesso.");

                return graph;
            }
            catch (Exception ex)
            {
                Log.Error("Nao foi possivel carregar o arquivo do modelo.\nVerifique o path para o arquivo .pb " +
                    $"com o argumento --graphFile: {ex.Message}");
                throw ex;
            }
        }
        private string[] LoadLabels(string modelDir, string labelFile)
        {
            try
            {
                var labels = File.ReadAllLines(Security.GrantAccess(Path.Join(modelDir, labelFile)));
                Log.Info($"Labels do modelo carregadas: {labels.ToString()}");
                return Labels;
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
        public List<ClassificationInference> Classify(string listPath)
        {
            try
            {
                List<ClassificationInference> Results = new List<ClassificationInference>();

                var list = JsonUtil<List<string>>.ReadJsonFile(listPath);

                foreach (var image in list)
                {
                    Log.Info($"Lendo imagem {image}");

                    NDArray imgArr = ReadTensorFromImageFile(Security.GrantAccess(Path.GetFullPath(image)));

                    using (var sess = tf.Session(Graph))
                        Results.Add(Predict(sess, imgArr, image));
                }
                return Results;
            }
            catch (FileNotFoundException ex)
            {
                Log.Error($"Arquivo não encontrado: {ex.Message}");
                throw;
            }
            catch (Exception ex)
            {
                Log.Error(ex.Message);
                throw;
            }
        }
        private ClassificationInference Predict(Session sess, NDArray imgArr, string image)
        {
            var input_operation = Graph.OperationByName(Input_name);
            var output_operation = Graph.OperationByName(Output_name);

            var classificationResults = new Dictionary<string, float>();

            Log.Info($"Executando classificação");

            var results = sess.run(output_operation.outputs[0], (input_operation.outputs[0], imgArr));
            results = np.squeeze(results);

            var argsort = results.argsort<float>();
            var top_k = argsort.Data<float>()
                .Skip(results.size - 5)
                .Reverse()
                .ToArray();

            foreach (float idx in top_k)
            {
                classificationResults.Add(Labels[(int)idx], results[(int)idx]);
            }

            Log.Info($"Imagem {Path.GetFileName(image)} classificada como {classificationResults.Keys.First()} com probabilidade de {classificationResults.Values.First()}");

            return new ClassificationInference()
            {
                Classifications = classificationResults,
                DateTime = DateTime.Now,
                ImagePath = image
            };
        }

        private NDArray ReadTensorFromImageFile(string file_name,
                                int input_height = 224,
                                int input_width = 224,
                                int input_mean = 117,
                                int input_std = 1)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.read_file(file_name, "file_reader");
            var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            var cast = tf.cast(decodeJpeg, tf.float32);
            var dims_expander = tf.expand_dims(cast, 0);
            var resize = tf.constant(new int[] { input_height, input_width });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            var sub = tf.subtract(bilinear, new float[] { input_mean });
            var normalized = tf.divide(sub, new float[] { input_std });

            using (var sess = tf.Session(graph))
                return sess.run(normalized);
        }
    }
}
