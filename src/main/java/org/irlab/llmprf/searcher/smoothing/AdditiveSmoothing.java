package org.irlab.llmprf.searcher.smoothing;

import org.irlab.llmprf.searcher.util.StatsProvider;

import java.util.concurrent.ConcurrentHashMap;

public final class AdditiveSmoothing extends AbstractSmoothing {

  private final ConcurrentHashMap<Integer, Long> cacheDocLength;
  private final ConcurrentHashMap<String, Long> cacheLexiconSize;

  public AdditiveSmoothing(double smoothingParameter, String docField, StatsProvider statsProvider) {

    super(smoothingParameter, docField, statsProvider);
    this.cacheDocLength = new ConcurrentHashMap<>();
    this.cacheLexiconSize = new ConcurrentHashMap<>();
  }

  @Override
  public double computeValue(String term, int doc) {
    final int termFreq = statsProvider.getTermFrequency(term, doc, docField);
    long docLength;
    long lexiconSize;

    if (cacheDocLength.containsKey(doc)) {
      docLength = cacheDocLength.get(doc);
    } else {
      docLength = statsProvider.getDocTokensSize(doc, docField);
      cacheDocLength.put(doc, docLength);
    }

    if (cacheLexiconSize.containsKey(docField)) {
      lexiconSize = cacheLexiconSize.get(docField);
    } else {
      lexiconSize = statsProvider.getCollectionLexiconSize(docField);
      cacheLexiconSize.put(docField, lexiconSize);
    }

    return (termFreq + smoothingParameter) / (docLength + smoothingParameter * lexiconSize);
  }

  @Override
  public double computeSpanSmoothedProb(String term, int spanTF, long spanLen) {
    long lexiconSize;
    if (cacheLexiconSize.containsKey(docField)) {
      lexiconSize = cacheLexiconSize.get(docField);
    } else {
      lexiconSize = statsProvider.getCollectionLexiconSize(docField);
      cacheLexiconSize.put(docField, lexiconSize);
    }
    return (spanTF + smoothingParameter) / (spanLen + smoothingParameter * lexiconSize);
  }
}