package org.irlab.llmprf.searcher.rf;

import org.irlab.llmprf.searcher.util.TermWeights;

import java.util.List;
import java.util.Map;

public interface RelevanceFeedback {

  TermWeights getTermWeights(final Map<Integer, Double> relevanceSet, final List<String> queryTerms);

}

