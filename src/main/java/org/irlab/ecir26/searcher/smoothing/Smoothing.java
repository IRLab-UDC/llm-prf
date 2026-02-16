package org.irlab.ecir26.searcher.smoothing;

import java.util.Set;

public interface Smoothing {

  double computeSmoothedProb(String term, int doc);

  Set<String> getDocTerms(int doc);

}