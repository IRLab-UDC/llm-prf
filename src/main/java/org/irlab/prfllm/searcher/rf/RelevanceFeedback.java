package org.irlab.prfllm.searcher.rf;
import java.util.Map;

import org.irlab.prfllm.searcher.util.TermWeights;

public interface RelevanceFeedback {

    TermWeights getTermWeights(final Map<Integer,Double> relevanceSet);
}

