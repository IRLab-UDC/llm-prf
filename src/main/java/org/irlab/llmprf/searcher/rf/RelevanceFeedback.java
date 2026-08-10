package org.irlab.llmprf.searcher.rf;

import org.irlab.llmprf.searcher.util.TermWeights;

import java.util.List;
import java.util.Map;
import java.util.Set;

public interface RelevanceFeedback {

  TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms);

  TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                             final Map<Integer, Set<String>> allowedTerms);

  TermWeights getTermWeightsFromSpans(final Map<Integer, Double> relevanceSet, final List<String> queryTerms,
                                      final Map<Integer, Map<String, Integer>> spanTermFreqs);

}
