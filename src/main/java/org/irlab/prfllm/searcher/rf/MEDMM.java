package org.irlab.prfllm.searcher.rf;

import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import org.apache.commons.lang3.mutable.MutableDouble;
import org.irlab.prfllm.searcher.smoothing.Smoothing;
import org.irlab.prfllm.searcher.util.TermWeights;

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
    // Get all vocab from docs in the relevance set.
    Set<String> vocab = getVocab(relevanceSet);

    Int2DoubleMap docWeights = getDocWeights(relevanceSet, queryTerms);

    TermWeights vocabWeights = new TermWeights();

    // For each term in the vocabulary, compute its weight.
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

  private Int2DoubleMap getDocWeights(Map<Integer, Double> relevanceSet, List<String> queryTerms) {
    Int2DoubleMap docWeights = new Int2DoubleOpenHashMap();
    MutableDouble weightsSum = new MutableDouble(0);

    relevanceSet.forEach((doc, ql) -> {
      // Compute Query Likelihood
      MutableDouble docql = new MutableDouble(0);
      for (String queryTerm : queryTerms) {
        // Ignore documents that are out of the vocabulary.
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