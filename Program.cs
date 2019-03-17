using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace SentimentAnalysis
{
    public static class ext
    {
        public static TrainCatalogBase.TrainTestData LoadData(this MLContext mlContext, string dataFilePath)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataFilePath, hasHeader: false);
            var splitDataView = mlContext.BinaryClassification.TrainTestSplit(dataView, testFraction: 0.2);
            return splitDataView;
        }
        public static ITransformer BuildAndTrainModel(this MLContext mlContext, IDataView splitTrainSet)
        {
            var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: DefaultColumnNames.Features, inputColumnName: nameof(SentimentData.SentimentText))
                            .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));            // Adds a FastTreeBinaryClassificationTrainer, the decision tree learner for this project  

            var model = pipeline.Fit(splitTrainSet);

            return model;
        }
        public static CalibratedBinaryClassificationMetrics Evaluate(this MLContext mlContext, ITransformer model, IDataView splitTestSet)
        {
            IDataView predictions = model.Transform(splitTestSet);

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            // The Accuracy metric gets the accuracy of a classifier, which is the proportion 
            // of correct predictions in the test set.

            // The Auc metric gets the area under the ROC curve.
            // The area under the ROC curve is equal to the probability that the classifier ranks
            // a randomly chosen positive instance higher than a randomly chosen negative one
            // (assuming 'positive' ranks higher than 'negative').

            // The F1Score metric gets the classifier's F1 score.
            // The F1 score is the harmonic mean of precision and recall:
            //  2 * precision * recall / (precision + recall).

            return metrics;
        }

        public static SentimentPrediction RunModel(this MLContext mlContext, ITransformer model, string sampleText)
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);

            return predictionFunction.Predict(new SentimentData { SentimentText = sampleText });
        }

        public static IEnumerable<SentimentPrediction> LoadAndRunModel(this MLContext mlContext, string modelFilePath, IEnumerable<string> sampleTexts)
        {
            var sentiments = sampleTexts.Select(x => new SentimentData { SentimentText = x });


            using (var stream = new FileStream(modelFilePath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var loadedModel = mlContext.Model.Load(stream);

                IDataView sentimentStreamingDataView = mlContext.Data.LoadFromEnumerable(sentiments);

                IDataView predictions = loadedModel.Transform(sentimentStreamingDataView);

                return mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            string dataFilePath = Path.Combine(Environment.CurrentDirectory, @"Data\yelp_labelled.txt");
            var splitDataView = mlContext.LoadData(dataFilePath);
            ITransformer model = mlContext.BuildAndTrainModel(splitDataView.TrainSet);
            var metrics = mlContext.Evaluate(model, splitDataView.TestSet);

            Console.WriteLine("\r\n--------------- Model quality metrics evaluation ---------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");

            string modelFilePaht = Path.Combine(Environment.CurrentDirectory, @"Data\Model.zip");
            using (var fs = new FileStream(modelFilePaht, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, fs);
            }

            var prediction = mlContext.RunModel(model, "This was a very bad steak");

            Console.WriteLine("\r\n--------------- Single Prediction with Trained Model ---------------");
            Console.WriteLine($"This was a very bad steak. Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

            var sampleTexts = new[]
            {
                "This was a horrible meal",
                "I love this spaghetti."
            };

            var predictions = mlContext.LoadAndRunModel(modelFilePaht, sampleTexts);
            var textsAndPredictions = sampleTexts.Zip(predictions, (text, predition) => (text, prediction));

            Console.WriteLine("\r\n--------------- Batch Prediction with Loaded Model ---------------");

            foreach (var pair in textsAndPredictions)
            {
                Console.WriteLine($"{pair.text} | Prediction: {(Convert.ToBoolean(pair.prediction.Prediction) ? "Positive" : "Negative")} | Probability: {pair.prediction.Probability} ");
            }

            string lineText = Console.ReadLine();
            while (lineText != "exit")
            {
                prediction = mlContext.RunModel(model, lineText);

                Console.WriteLine($"{(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
                lineText = Console.ReadLine();
            }
        }
    }
}