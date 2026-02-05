package org.irlab.prfllm.searcher.smoothing;

import org.irlab.prfllm.searcher.util.StatsProvider;

import java.util.concurrent.ConcurrentHashMap;

public final class DirichletSmoothing extends AbstractSmoothing {

  private final ConcurrentHashMap<Integer, Long> cacheDocLength;

  public DirichletSmoothing(double smoothingParameter, String docField, StatsProvider statsProvider) {

    super(smoothingParameter, docField, statsProvider);
    this.cacheDocLength = new ConcurrentHashMap<>();
  }

  public DirichletSmoothing(double smoothingParameter, String docField, StatsProvider statsProvider,
                            StatsProvider statsProviderBackground) {

    super(smoothingParameter, docField, statsProvider, statsProviderBackground);
    this.cacheDocLength = new ConcurrentHashMap<>();
  }

  @Override
  public double computeValue(String term, int doc) {

    final int termFreq = statsProvider.getTermFrequency(term, doc, docField);
    final double backgroundProb = computeBackgroundProb(term);
    long docLength;

    if (cacheDocLength.containsKey(doc)) {
      docLength = cacheDocLength.get(doc);
    } else {
      docLength = statsProvider.getDocTokensSize(doc, docField);
      cacheDocLength.put(doc, docLength);
    }

    return (termFreq + smoothingParameter * backgroundProb) / (docLength + smoothingParameter);
  }

  @Override
  public String getName() {

    return String.format("Dirichlet-mu-%1.2f", smoothingParameter);
  }
}