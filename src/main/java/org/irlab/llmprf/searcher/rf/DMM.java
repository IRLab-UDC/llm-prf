package org.irlab.llmprf.searcher.rf;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.irlab.llmprf.searcher.smoothing.Smoothing;
import org.irlab.llmprf.searcher.util.TermWeights;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class DMM extends AbstractRelevanceFeedback {

  private final double lambda;

  public DMM(String docField, Smoothing smoothing, double lambda) {
    super(docField, smoothing);
    this.lambda = lambda;
  }

  @Override
  public TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms) {
    return getTermWeights(relevanceSet, queryTerms, null);
  }

  @Override
  public TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                                    final Map<Integer, Set<String>> allowedTerms) {
    final int relevanceSetSize = relevanceSet.size();
    Set<String> vocab = getVocab(relevanceSet, allowedTerms);

    TermWeights vocabWeights = new TermWeights();

    for (String term : vocab) {
      final MutableDouble sum = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> sum.add(Math.log(smoothing.computeSmoothedProb(term, doc))));

      final double backgroundProbLog = Math.log(smoothing.computeBackgroundProb(term));
      final double pwf = ((sum.doubleValue() / relevanceSetSize) - (lambda * backgroundProbLog)) / (1 - lambda);

      vocabWeights.setTermWeight(term, Math.exp(pwf));
    }

    return vocabWeights;
  }

  @Override
  public TermWeights getTermWeightsFromSpans(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                                              final Map<Integer, Map<String, Integer>> spanTermFreqs) {
    final int relevanceSetSize = relevanceSet.size();
    Set<String> vocab = getVocabFromSpans(relevanceSet, spanTermFreqs);
    Map<Integer, Long> spanLen = spanLengths(spanTermFreqs);

    TermWeights vocabWeights = new TermWeights();

    for (String term : vocab) {
      final MutableDouble sum = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> {
        Map<String, Integer> tf = spanTermFreqs.get(doc);
        int stf = (tf != null) ? tf.getOrDefault(term, 0) : 0;
        long slen = spanLen.getOrDefault(doc, 1L);
        sum.add(Math.log(smoothing.computeSpanSmoothedProb(term, stf, slen)));
      });

      final double backgroundProbLog = Math.log(smoothing.computeBackgroundProb(term));
      final double pwf = ((sum.doubleValue() / relevanceSetSize) - (lambda * backgroundProbLog)) / (1 - lambda);

      vocabWeights.setTermWeight(term, Math.exp(pwf));
    }

    return vocabWeights;
  }
}
