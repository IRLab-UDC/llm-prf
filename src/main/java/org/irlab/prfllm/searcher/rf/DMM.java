package org.irlab.prfllm.searcher.rf;

import org.apache.commons.lang3.mutable.MutableDouble;
import org.irlab.prfllm.searcher.smoothing.Smoothing;
import org.irlab.prfllm.searcher.util.TermWeights;

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
    final int relevanceSetSize = relevanceSet.size();

    // Get all vocab from docs in the relevance set.
    Set<String> vocab = getVocab(relevanceSet);

    TermWeights vocabWeights = new TermWeights();

    // For each term in the vocabulary, compute its weight.
    for (String term : vocab) {
      final MutableDouble sum = new MutableDouble(0);

      // Sum log probabilities across relevant documents
      relevanceSet.forEach((doc, ql) -> {
        double docProb = smoothing.computeSmoothedProb(term, doc);
        sum.add(Math.log(docProb));
      });

      // Compute background probability
      final double backgroundProbLog = Math.log(smoothing.computeBackgroundProb(term));

      // DMM formula: ((sum / |R|) - (lambda * log P(t|C))) / (1 - lambda)
      final double pwf = ((sum.doubleValue() / relevanceSetSize) - (lambda * backgroundProbLog)) / (1 - lambda);

      vocabWeights.setTermWeight(term, Math.exp(pwf));
    }

    return vocabWeights;
  }
}
