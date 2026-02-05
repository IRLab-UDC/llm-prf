package org.irlab.prfllm.searcher.rf;

import org.irlab.prfllm.searcher.util.TermWeights;

import java.util.Map;

public interface RelevanceFeedback {

  TermWeights getTermWeights(final Map<Integer, Double> relevanceSet);
}

