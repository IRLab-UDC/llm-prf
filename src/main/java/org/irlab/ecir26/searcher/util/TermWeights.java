package org.irlab.ecir26.searcher.util;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import it.unimi.dsi.fastutil.objects.ObjectSet;

import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

public final class TermWeights {

  private Map<String, Double> weights;

  public TermWeights() {

    weights = new Object2DoubleOpenHashMap<>();
  }

  public static TermWeights fromTerms(List<String> terms) {

    TermWeights f = new TermWeights();
    terms.forEach(term -> f.addTermWeight(term, 1.0f));

    return f;
  }

  public static TermWeights interpolate(TermWeights original, TermWeights expand, double originalWeight) {

    ObjectSet<String> mixedTerms = new ObjectOpenHashSet<>();

    mixedTerms.addAll(original.weights.keySet());
    mixedTerms.addAll(expand.weights.keySet());

    TermWeights inteporlatedWeights = new TermWeights();

    mixedTerms.forEach(term -> {

      double weight = (original.getTermWeight(term) * originalWeight)
                      + (1 - originalWeight) * expand.getTermWeight(term);
      inteporlatedWeights.addTermWeight(term, weight);
    });

    return inteporlatedWeights;
  }

  public void addTermWeight(String term, double weight) {
    weights.merge(term, weight, Double::sum);
  }

  public double getTermWeight(String term) {
    return weights.getOrDefault(term, 0.0d);
  }

  public TermWeights scaleToL1Norm() {
    final double norm = getL1Norm();
    weights.keySet().forEach(term -> weights.put(term, weights.get(term) / norm));
    return this;
  }

  public double getL1Norm() {
    // The L1 norm is the sum of the absolute values of the weights
    return weights.values().stream().mapToDouble(Math::abs).sum();
  }

  public void forEach(final BiConsumer<String, Double> consumer) {
    weights.forEach(consumer);
  }

  public TermWeights pruneToSize(int size) {

    Map<String, Double> sortedWeights = PutHashMapInOrder.sort(weights);

    Object2DoubleMap<String> pruned = new Object2DoubleOpenHashMap<>();
    sortedWeights.keySet()
                 .stream()
                 .takeWhile(term -> pruned.size() < size)
                 .forEach(term -> pruned.put(term, sortedWeights.get(term).doubleValue()));

    this.weights = pruned;
    return this;
  }

  @Override
  public String toString() {
    StringBuilder string = new StringBuilder();

    string.append("{");
    weights.forEach((term, weight) -> string.append(String.format("[term: %s, weight: %1.3f],", term, weight)));
    string.append("}");

    return string.toString();
  }
}
