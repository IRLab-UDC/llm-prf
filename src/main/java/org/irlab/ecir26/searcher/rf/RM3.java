package org.irlab.ecir26.searcher.rf;


import org.apache.commons.lang3.mutable.MutableDouble;
import org.irlab.ecir26.searcher.smoothing.Smoothing;
import org.irlab.ecir26.searcher.util.TermWeights;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class RM3 extends AbstractRelevanceFeedback {

  public RM3(String docField, Smoothing documentSmoothing) {

    super(docField, documentSmoothing);
  }

  @Override
  public TermWeights getTermWeights(Map<Integer, Double> relevanceSet) {

    return estimateWeights(relevanceSet);
  }


  protected TermWeights estimateWeights(Map<Integer, Double> relevanceSet) {

    Set<String> vocab = new HashSet<>();

    relevanceSet.forEach((doc, ql) -> {

      Set<String> docTerms = documentSmoothing.getDocTerms(doc);
      vocab.addAll(docTerms);
    });
    TermWeights vocabWeights = new TermWeights();

    for (String term : vocab) {

      MutableDouble pwr = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> pwr.add(computeTermDocWeight(term, doc, ql)));
      vocabWeights.addTermWeight(term, pwr.doubleValue());
    }

    return vocabWeights;
  }

  private double computeTermDocWeight(final String term, final int doc, double queryLikelihood) {

    double pwd = documentSmoothing.computeSmoothedProb(term, doc);
    return Math.exp(Math.log(pwd) + queryLikelihood);
  }

  @Override
  protected String getName() {

    return String.format("RM3-docsmoothing-%s", documentSmoothing);
  }
}
