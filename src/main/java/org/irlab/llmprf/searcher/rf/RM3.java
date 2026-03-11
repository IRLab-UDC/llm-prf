package org.irlab.llmprf.searcher.rf;


import org.apache.commons.lang3.mutable.MutableDouble;
import org.irlab.llmprf.searcher.smoothing.Smoothing;
import org.irlab.llmprf.searcher.util.TermWeights;

import java.util.List;
import java.util.Map;
import java.util.Set;

// Name should be RM1 since it is more accurate. Interpolation is made outside this class.
public final class RM3 extends AbstractRelevanceFeedback {

  public RM3(String docField, Smoothing smoothing) {
    super(docField, smoothing);
  }

  @Override
  public TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms) {
    // Get all vocab from docs in the relevance set.
    Set<String> vocab = getVocab(relevanceSet);

    // Compute weight for each term.
    TermWeights vocabWeights = new TermWeights();
    for (String term : vocab) {
      MutableDouble pwr = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> pwr.add(computeTermDocWeight(term, doc, ql)));
      vocabWeights.setTermWeight(term, pwr.doubleValue());
    }

    return vocabWeights;
  }

  private double computeTermDocWeight(final String term, final int doc, double queryLikelihood) {
    double pwd = smoothing.computeSmoothedProb(term, doc);
    return Math.exp(Math.log(pwd) + queryLikelihood);
  }
}
