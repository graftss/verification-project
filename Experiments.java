import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.BayesianLogisticRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MLPClassifier;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.REPTree;

import java.io.File;
import java.util.Enumeration;
import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import java.util.stream.Collectors;

class Experiments {
  static final String datasetPath = "./combined.arff";

  public static void main(String[] args) throws Exception {
    List<ExperimentResult> results = new ArrayList<ExperimentResult>();
    Classifier classifier;
    Evaluation eval;
    Instances dataset;
    SplitDataset data;

    results.addAll(runFullDatasetExperiments());
    results.addAll(runSingleProjectTrainedExperiments());

    printResults(results);
  }

  public static void printResults(List<ExperimentResult> results) {
    System.out.println("Experiment\tClassifier\tTP\tFN\tFP\tTN");
    results.forEach(r -> r.print());
  }

  public static List<ExperimentResult> runFullDatasetExperiments() {
    List<Classifier> classifiers = new ArrayList<Classifier>();
    List<ExperimentResult> results = new ArrayList<ExperimentResult>();
    // classifiers.add(new LibSVM()); // can't find one of the dependencies online
    // classifiers.add(new ClassificationViaRegression()); // doesn't work

    classifiers.add(new NaiveBayes());
    classifiers.add(new Logistic());
    classifiers.add(new MLPClassifier());
    classifiers.add(new SGD());
    classifiers.add(new SMO());
    classifiers.add(new VotedPerceptron());
    classifiers.add(new AttributeSelectedClassifier());
    classifiers.add(new LogitBoost());
    classifiers.add(new DecisionStump());
    classifiers.add(new RandomForest());
    classifiers.add(new RandomTree());
    classifiers.add(new REPTree());

    classifiers.forEach(classifier -> {
      Evaluation eval = runCrossValidationExperiment(classifier);
      results.add(new ExperimentResult(eval, classifier, "10-fold cross validation"));
    });

    return results;
  }

  public static List<ExperimentResult> runSingleProjectTrainedExperiments() {
    List<ExperimentResult> results = new ArrayList<ExperimentResult>();
    Classifier classifier = new Logistic();

    (new Dataset(datasetPath)).projectNames().forEach(name -> {
      SplitDataset data = splitDatasetOnProjectName(name);
      Evaluation eval = runExperiment(classifier, data);
      String label = String.format("Trained on %s", name);

      results.add(new ExperimentResult(eval, classifier, label));
    });

    return results;
  }

  public static SplitDataset splitDatasetOnProjectName(String projectName) {
    Dataset dataset = new Dataset(datasetPath);
    Instances training = dataset.emptyInstances();
    Instances validation = dataset.emptyInstances();

    // separate the dataset into training and validation subsets, based
    // on the value of the 0-index attribute (which is project name)
    Instance instance;
    while ((instance = dataset.getNextInstance()) != null) {
      validation.add(instance);

      if (instance.stringValue(0).equals(projectName)) {
        training.add(instance);
      }
    }

    // delete project name attributes
    training.deleteAttributeAt(0);
    validation.deleteAttributeAt(0);

    return new SplitDataset(training, validation);
  }

  // evaluates the given classifier using 10 times 10-fold cross-validation
  public static Evaluation runCrossValidationExperiment(Classifier classifier) {
    try {
      Instances data = (new Dataset(datasetPath)).fullDataset();
      // delete project name attribute
      data.deleteAttributeAt(0);
      Evaluation eval = new Evaluation(data);
      eval.crossValidateModel(classifier, data, 10, new Random(1));
      return eval;
    } catch (Exception e) {
      System.out.println("error running cross validation experiment");
      System.out.println(e);
      return null;
    }
  }

  // evaluates the given classifier using the given training and evaluation datasets
  public static Evaluation runExperiment(Classifier classifier, SplitDataset data) {
    try {
      classifier.buildClassifier(data.training);
      Evaluation eval = new Evaluation(data.training);
      eval.evaluateModel(classifier, data.validation);
      return eval;
    } catch (Exception e) {
      System.out.println("error running experiment");
      System.out.println(e);
      return null;
    }
  }

  // responsible for loading and fetching instances from our ARFF dataset
  static class Dataset {
    public ArffLoader loader;
    public Instances structure;

    public Dataset(String path) {
      try {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(path));
        Instances structure = loader.getStructure();
        structure.setClassIndex(structure.numAttributes() - 1);

        this.loader = loader;
        this.structure = structure;
      } catch (Exception e) {}
    }

    public Instance getNextInstance() {
      try {
        return loader.getNextInstance(structure);
      } catch (Exception e) {
        return null;
      }
    }

    public Instances emptyInstances() {
      return new Instances(structure);
    }

    public Instances fullDataset() {
      try {
        Instances dataset = loader.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        return dataset;
      } catch (Exception e) {
        return null;
      }
    }

    public List<String> projectNames() {
      List<String> result = new ArrayList<String>();

      Enumeration<Object> e = structure.attribute(0).enumerateValues();
      while (e.hasMoreElements()) {
        result.add(e.nextElement().toString());
      }

      return result;
    }
  }

  static class SplitDataset {
    public Instances training;
    public Instances validation;

    public SplitDataset(Instances training, Instances validation) {
      this.training = training;
      this.validation = validation;
    }
  }

  static class ExperimentResult {
    public Evaluation eval;
    public Classifier classifier;
    public String label;

    public ExperimentResult(Evaluation eval, Classifier classifier, String label) {
      this.eval = eval;
      this.classifier = classifier;
      this.label = label;
    }

    public String evalString() {
      double[][] m = eval.confusionMatrix();
      double tp = m[0][0];
      double fp = m[0][1];
      double fn = m[1][0];
      double tn = m[1][1];

      return String.format(
        "%d\t%d\t%d\t%d",
        (int)m[0][0], // true positives
        (int)m[0][1], // false negatives
        (int)m[1][0], // false positives
        (int)m[1][1] // true negatives
      );
    }

    public void print() {
      System.out.println(String.format(
        "%s\t%s\t%s",
        label,
        classifier.getClass().getName(),
        evalString()
      ));
    }
  }
}
