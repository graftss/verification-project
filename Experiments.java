import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.Classifier;

import java.io.File;
import java.util.Enumeration;
import java.util.Random;

class Experiments {
  static final String datasetPath = "./combined.arff";

  public static void main(String[] args) throws Exception {
    SplitDataset data = splitDatasetOnProjectName("Berek");
    J48 classifier = new J48();

    // runExperiment(classifier, data);
    // runCrossValidationExperiment(classifier);
  }

  public static SplitDataset splitDatasetOnProjectName(String projectName) {
    Dataset dataset = new Dataset(datasetPath);
    Instances training = dataset.emptyInstances();
    Instances validation = dataset.emptyInstances();

    // separate the dataset into training and validation subsets, based
    // on the value of the 0-index attribute (which is project name)
    Instances nextTarget;
    Instance instance;
    while ((instance = dataset.getNextInstance()) != null) {
      nextTarget = instance.stringValue(0).equals(projectName) ? training : validation;
      nextTarget.add(instance);
    }

    return new SplitDataset(training, validation);
  }

  // evaluates the given classifier using 10 times 10-fold cross-validation
  public static void runCrossValidationExperiment(Classifier classifier, Instances data) throws Exception {
    Evaluation eval = new Evaluation(data);
    eval.crossValidateModel(classifier, data, 10, new Random(1));
  }

  // evaluates the given classifier using the given training and evaluation datasets
  public static void runExperiment(Classifier classifier, SplitDataset data) throws Exception {
    classifier.buildClassifier(data.training);
    Evaluation eval = new Evaluation(data.training);
    eval.evaluateModel(classifier, data.validation);
    System.out.println(eval.toSummaryString("\nResults\n======\n", false));
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
  }

  static class SplitDataset {
    public Instances training;
    public Instances validation;

    public SplitDataset(Instances training, Instances validation) {
      this.training = training;
      this.validation = validation;
    }
  }
}
