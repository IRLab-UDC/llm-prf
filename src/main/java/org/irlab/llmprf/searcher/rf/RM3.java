package org.irlab.llmprf.searcher.rf;


import org.apache.commons.lang3.mutable.MutableDouble;
import org.irlab.llmprf.searcher.smoothing.Smoothing;
import org.irlab.llmprf.searcher.util.TermWeights;

import java.util.List;
import java.util.Map;
import java.util.Set;

public final class RM3 extends AbstractRelevanceFeedback {

  public RM3(String docField, Smoothing smoothing) {
    super(docField, smoothing);
  }

  @Override
  public TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms) {
    return getTermWeights(relevanceSet, queryTerms, null);
  }

  @Override
  public TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                                    final Map<Integer, Set<String>> allowedTerms) {
    Set<String> vocab = getVocab(relevanceSet, allowedTerms);

    TermWeights vocabWeights = new TermWeights();
    for (String term : vocab) {
      MutableDouble pwr = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> pwr.add(computeTermDocWeight(term, doc, ql)));
      vocabWeights.setTermWeight(term, pwr.doubleValue());
    }

    return vocabWeights;
  }

  @Override
  public TermWeights getTermWeightsFromSpans(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                                              final Map<Integer, Map<String, Integer>> spanTermFreqs) {
    Set<String> vocab = getVocabFromSpans(relevanceSet, spanTermFreqs);
    Map<Integer, Long> spanLen = spanLengths(spanTermFreqs);

    TermWeights vocabWeights = new TermWeights();
    for (String term : vocab) {
      MutableDouble pwr = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> {
        Map<String, Integer> tf = spanTermFreqs.get(doc);
        int stf = (tf != null) ? tf.getOrDefault(term, 0) : 0;
        long slen = spanLen.getOrDefault(doc, 1L);
        double pwd = smoothing.computeSpanSmoothedProb(term, stf, slen);
        pwr.add(Math.exp(Math.log(pwd) + ql));
      });
      vocabWeights.setTermWeight(term, pwr.doubleValue());
    }

    return vocabWeights;
  }

  private double computeTermDocWeight(final String term, final int doc, double queryLikelihood) {
    double pwd = smoothing.computeSmoothedProb(term, doc);
    return Math.exp(Math.log(pwd) + queryLikelihood);
  }
}
