package org.irlab.prfllm.searcher.smoothing;

import java.util.Set;

public interface Smoothing {

  double computeSmoothedProb(String term, int doc);

  double computeMLE(String term, int doc);

  double computeBackgroundProb(String term);

  Set<String> getDocTerms(int doc);

  boolean termExists(String term);
}