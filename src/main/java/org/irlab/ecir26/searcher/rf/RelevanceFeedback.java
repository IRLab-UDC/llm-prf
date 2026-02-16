package org.irlab.ecir26.searcher.rf;

import org.irlab.ecir26.searcher.util.TermWeights;

import java.util.Map;

public interface RelevanceFeedback {

  TermWeights getTermWeights(final Map<Integer, Double> relevanceSet);
}

