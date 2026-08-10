package org.irlab.llmprf.searcher.rf;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import org.apache.commons.lang3.mutable.MutableDouble;
import org.irlab.llmprf.searcher.smoothing.Smoothing;
import org.irlab.llmprf.searcher.util.TermWeights;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class MEDMM extends AbstractRelevanceFeedback {

  private final double lambda;
  private final double beta;

  public MEDMM(String docField, Smoothing documentSmoothing, double lambda, double beta) {
    super(docField, documentSmoothing);
    this.beta = beta;
    this.lambda = lambda;
  }

  @Override
  public TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms) {
    return getTermWeights(relevanceSet, queryTerms, null);
  }

  @Override
  public TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                                    final Map<Integer, Set<String>> allowedTerms) {
    Set<String> vocab = getVocab(relevanceSet, allowedTerms);
    Int2DoubleMap docWeights = getDocWeights(relevanceSet, queryTerms);

    TermWeights vocabWeights = new TermWeights();

    for (String term : vocab) {
      final MutableDouble sum = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> {
        double docProb = smoothing.computeSmoothedProb(term, doc);
        sum.add(Math.log(docProb) * docWeights.get(doc.intValue()));
      });

      final double backgroundProbLog = Math.log(smoothing.computeBackgroundProb(term));
      final double pwf = (sum.doubleValue() - (lambda * backgroundProbLog)) / beta;

      vocabWeights.setTermWeight(term, Math.exp(pwf));
    }

    return vocabWeights;
  }

  @Override
  public TermWeights getTermWeightsFromSpans(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                                              final Map<Integer, Map<String, Integer>> spanTermFreqs) {
    Set<String> vocab = getVocabFromSpans(relevanceSet, spanTermFreqs);
    Map<Integer, Long> spanLen = spanLengths(spanTermFreqs);
    Int2DoubleMap docWeights = getDocWeights(relevanceSet, queryTerms);

    TermWeights vocabWeights = new TermWeights();

    for (String term : vocab) {
      final MutableDouble sum = new MutableDouble(0);
      relevanceSet.forEach((doc, ql) -> {
        Map<String, Integer> tf = spanTermFreqs.get(doc);
        int stf = (tf != null) ? tf.getOrDefault(term, 0) : 0;
        long slen = spanLen.getOrDefault(doc, 1L);
        double spanProb = smoothing.computeSpanSmoothedProb(term, stf, slen);
        sum.add(Math.log(spanProb) * docWeights.get(doc.intValue()));
      });

      final double backgroundProbLog = Math.log(smoothing.computeBackgroundProb(term));
      final double pwf = (sum.doubleValue() - (lambda * backgroundProbLog)) / beta;

      vocabWeights.setTermWeight(term, Math.exp(pwf));
    }

    return vocabWeights;
  }

  private Int2DoubleMap getDocWeights(Map<Integer, Double> relevanceSet, List<String> queryTerms) {
    Int2DoubleMap docWeights = new Int2DoubleOpenHashMap();
    MutableDouble weightsSum = new MutableDouble(0);

    relevanceSet.forEach((doc, ql) -> {
      MutableDouble docql = new MutableDouble(0);
      for (String queryTerm : queryTerms) {
        if (smoothing.termExists(queryTerm)) {
          docql.add(Math.log(smoothing.computeSmoothedProb(queryTerm, doc)));
        }
      }

      final double weight = Math.exp(docql.doubleValue());
      docWeights.put(doc.intValue(), weight);
      weightsSum.add(weight);
    });

    docWeights.replaceAll((doc, weight) -> weight / weightsSum.doubleValue());

    return docWeights;
  }

}