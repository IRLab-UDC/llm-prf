package org.irlab.prfllm.searcher.smoothing;

import java.util.Set;

public interface Smoothing {

  double computeSmoothedProb(String term, int doc);

  Set<String> getDocTerms(int doc);

  double computeBackgroundProb(String term);

  boolean termExists(String term);
}