using System;
using System.IO;
using Newtonsoft.Json;

namespace MachineLearningToolkit.ImageClassification.Utility
{
    public class JsonUtil<TEntity>
    {
        public static TEntity StringJsonToObject(string json)
        {
            try
            {
                return JsonConvert.DeserializeObject<TEntity>(json);
            }
            catch (Exception)
            {
                return default(TEntity);
            }
        }

        public static string ObjectToJson(TEntity entity)
        {
            try
            {
                return JsonConvert.SerializeObject(entity);
            }
            catch (Exception)
            {
                throw;
            }
        }

        public static bool WriteJsonOnFile(TEntity entity, string path)
        {
            try
            {
                var json = ObjectToJson(entity);
                File.WriteAllText(path, json);
                return true;
            }
            catch (Exception)
            {
                throw;
            }
        }

        public static TEntity ReadJsonFile(string jsonFile)
        {
            try
            {
                string stringJson = File.ReadAllText(jsonFile);
                return StringJsonToObject(stringJson);
            }
            catch (Exception)
            {
                throw;
            }
        }

        public static bool IsValidJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                return false;
            }

            if (json.StartsWith("{") || json.StartsWith("[") && json.EndsWith("}") || json.EndsWith("]"))
            {
                return true;
            }

            try
            {
                var obj = JsonConvert.DeserializeObject(json);
                return obj.GetType() == typeof(TEntity);

            }
            catch (JsonReaderException)
            {
                return false;
            }
        }
    }
}
