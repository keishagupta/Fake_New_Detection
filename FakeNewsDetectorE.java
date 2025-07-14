
import java.io.*;
import java.util.*;

public class FakeNewsDetectorE {
    private static final String DATASET_PATH = "C:\\Users\\Lenovo\\Downloads\\FAKENEWSDETECTION\\fake_or_real_news.csv";

    // Inner class to store news data
    static class NewsData {
        String title;
        String text;
        String label;

        NewsData(String title, String text, String label) {
            this.title = title;
            this.text = text;
            this.label = label.equalsIgnoreCase("REAL") || label.equalsIgnoreCase("FAKE") ? label : "UNKNOWN";
        }
    }

    public static List<NewsData> loadDataset(String filePath) throws IOException {
        List<NewsData> dataset = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            boolean firstLine = true;
            while ((line = br.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                String[] parts = line.split(",(?=(?:[^"]*\"[^"]*\")*[^"]*$)", -1);
                if (parts.length >= 3) {
                    dataset.add(new NewsData(parts[0].trim(), parts[1].trim(), parts[2].trim()));
                }
            }
        }
        return dataset;
    }

    public static void appendToDataset(String filePath, String title, String text, String label) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filePath, true))) {
            bw.write(String.format("\"%s\",\"%s\",\"%s\"", title, text, label));
            bw.newLine();
        }
    }

    private static String[] normalizeText(String text) {
        return text.toLowerCase().replaceAll("[^a-zA-Z0-9]", " ").split("\\s+");
    }

    static class TermFrequencyVectorizer {
        public static Map<String, Integer> buildVocabulary(List<NewsData> dataset) {
            Map<String, Integer> vocabulary = new HashMap<>();
            int index = 0;
            for (NewsData news : dataset) {
                String[] words = normalizeText(news.title + " " + news.text);
                for (String word : words) {
                    if (!word.isEmpty() && !vocabulary.containsKey(word)) {
                        vocabulary.put(word, index++);
                    }
                }
            }
            return vocabulary;
        }

        public static Map<String, Integer> transformToBagOfWords(String text, Map<String, Integer> vocabulary) {
            Map<String, Integer> bagOfWords = new HashMap<>();
            String[] words = normalizeText(text);
            for (String word : vocabulary.keySet()) {
                bagOfWords.put(word, 0);
            }
            for (String word : words) {
                if (vocabulary.containsKey(word)) {
                    bagOfWords.put(word, 1);
                }
            }
            return bagOfWords;
        }
    }

    static class NaiveBayesClassifier {
        private Map<String, Double> classProbabilities;
        private Map<String, Map<String, Integer>> wordCounts;
        private Map<String, Integer> classCounts;
        private int vocabularySize;

        public void train(List<NewsData> dataset, Map<String, Integer> vocabulary) {
            classProbabilities = new HashMap<>();
            wordCounts = new HashMap<>();
            classCounts = new HashMap<>();
            vocabularySize = vocabulary.size();
            for (NewsData news : dataset) {
                classCounts.put(news.label, classCounts.getOrDefault(news.label, 0) + 1);
                wordCounts.putIfAbsent(news.label, new HashMap<>());
                Map<String, Integer> termFrequency = TermFrequencyVectorizer.transformToBagOfWords(news.title + " " + news.text, vocabulary);
                for (String word : termFrequency.keySet()) {
                    wordCounts.get(news.label).put(word, wordCounts.get(news.label).getOrDefault(word, 0) + termFrequency.get(word));
                }
            }
            for (String label : classCounts.keySet()) {
                classProbabilities.put(label, (double) classCounts.get(label) / dataset.size());
            }
        }

        public String predict(String input, Map<String, Integer> vocabulary) {
            Map<String, Integer> vector = TermFrequencyVectorizer.transformToBagOfWords(input, vocabulary);
            Map<String, Double> logProbabilities = new HashMap<>();
            for (String label : classCounts.keySet()) {
                double logProb = Math.log(classProbabilities.get(label));
                for (String word : vector.keySet()) {
                    int wordCount = wordCounts.getOrDefault(label, new HashMap<>()).getOrDefault(word, 0);
                    logProb += Math.log((double) (wordCount + 1) / (classCounts.get(label) + vocabularySize));
                }
                logProbabilities.put(label, logProb);
            }
            return logProbabilities.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        }
    }

    private static void startUserInteraction(Scanner scanner, NaiveBayesClassifier classifier, List<NewsData> dataset, Map<String, Integer> vocabulary) {
        System.out.println("\nWelcome to the Fake News Detector!\n");
        System.out.flush();
        int correctPredictions = 0;
        int totalPredictions = 0;

        while (true) {
            System.out.print("Enter news title (or 'exit' to quit): ");
            System.out.flush();
            String title = scanner.nextLine();
            if (title.equalsIgnoreCase("exit")) {
                break;
            }
            System.out.print("Enter news text: ");
            System.out.flush();
            String text = scanner.nextLine();

            String prediction = classifier.predict(title + " " + text, vocabulary);
            if (!prediction.equals("REAL") && !prediction.equals("FAKE")) {
                System.out.println("Unknown prediction. Defaulting to FAKE.");
                prediction = "FAKE";
            }

            System.out.println("Predicted Label: " + prediction);
            System.out.print("Enter the actual label (REAL/FAKE): ");
            System.out.flush();
            String actualLabel = scanner.nextLine().toUpperCase();

            if (!actualLabel.equals("REAL") && !actualLabel.equals("FAKE")) {
                System.out.println("Invalid label entered. Please enter either 'REAL' or 'FAKE'.");
            } else {
                totalPredictions++;
                if (prediction.equals(actualLabel)) {
                    correctPredictions++;
                }
            }

            double accuracy = (double) correctPredictions / totalPredictions;
            System.out.printf("Dynamic Accuracy: %.2f%%%n", accuracy * 100);

            System.out.print("Do you want to add this news to the dataset? (yes/no): ");
            System.out.flush();
            if (scanner.nextLine().equalsIgnoreCase("yes")) {
                try {
                    appendToDataset(DATASET_PATH, title, text, prediction);
                    System.out.println("News added to the dataset.");
                } catch (IOException e) {
                    System.out.println("Error adding news to dataset: " + e.getMessage());
                }
            }
        }
        System.out.println("Exiting the application.");
    }

    private static void evaluateClassifier(NaiveBayesClassifier classifier, List<NewsData> testSet, Map<String, Integer> vocabulary) {
        int correctPredictions = 0;
        int totalPredictions = testSet.size();
        for (NewsData news : testSet) {
            String prediction = classifier.predict(news.title + " " + news.text, vocabulary);
            if (prediction.equals(news.label)) {
                correctPredictions++;
            }
        }
        double accuracy = (double) correctPredictions / totalPredictions;
        System.out.printf("Evaluation Accuracy: %.2f%%%n", accuracy * 100);
    }

    public static void main(String[] args) {
        try {
            List<NewsData> dataset = loadDataset(DATASET_PATH);
            System.out.println("Dataset loaded: " + dataset.size() + " records");
            Map<String, Integer> vocabulary = TermFrequencyVectorizer.buildVocabulary(dataset);

            Collections.shuffle(dataset);
            int trainSize = (int) (0.70 * dataset.size());
            List<NewsData> trainSet = dataset.subList(0, trainSize);
            List<NewsData> testSet = dataset.subList(trainSize, dataset.size());

            NaiveBayesClassifier naiveBayesClassifier = new NaiveBayesClassifier();
            naiveBayesClassifier.train(trainSet, vocabulary);

            Scanner scanner = new Scanner(System.in);
            startUserInteraction(scanner, naiveBayesClassifier, trainSet, vocabulary);

            evaluateClassifier(naiveBayesClassifier, testSet, vocabulary);
        } catch (IOException e) {
            System.out.println("Error loading dataset: " + e.getMessage());
        }
    }
}