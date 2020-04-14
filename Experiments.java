import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

import java.io.File;

class Experiments {
  static final String dataPath = "./combined.arff";

  public static void main(String[] args) {
    Instances data = loadData();
    System.out.println(data);
  }

  public static Instances loadData() {
    try {
      ArffLoader loader = new ArffLoader();
      loader.setFile(new File(dataPath));
      Instances structure = loader.getStructure();

      // make the last attribute ("bug") be the class of the data
      structure.setClassIndex(structure.numAttributes() - 1);

      return structure;
    } catch (Exception e) {
      return null;
    }
  }
}
