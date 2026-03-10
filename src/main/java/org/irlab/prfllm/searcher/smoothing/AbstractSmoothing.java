package org.irlab.prfllm.searcher.smoothing;

import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.irlab.prfllm.searcher.util.StatsProvider;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public abstract class AbstractSmoothing implements Smoothing {

  protected final double smoothingParameter;
  protected final String docField;
  protected final StatsProvider statsProvider;
  private final ConcurrentHashMap<Pair<String, Integer>, Double> cacheSmoothed;
  private final ConcurrentHashMap<Integer, Set<String>> cacheTerms;
  private final ConcurrentHashMap<String, Double> cacheBackground;


  public AbstractSmoothing(double smoothingParameter, String docField, StatsProvider statsProvider) {
    this.smoothingParameter = smoothingParameter;
    this.docField = docField;
    this.statsProvider = statsProvider;
    this.cacheSmoothed = new ConcurrentHashMap<>();
    this.cacheTerms = new ConcurrentHashMap<>();
    this.cacheBackground = new ConcurrentHashMap<>();
  }

  protected abstract double computeValue(String term, int doc);

  @Override
  public double computeSmoothedProb(String term, int doc) {
    Pair<String, Integer> key = Pair.of(term, doc);

    if (cacheSmoothed.containsKey(key)) {
      return cacheSmoothed.get(key);
    }

    double value = computeValue(term, doc);

    cacheSmoothed.put(key, value);
    return value;
  }


  @Override
  public Set<String> getDocTerms(int doc) {
    if (cacheTerms.containsKey(doc)) {
      return cacheTerms.get(doc);
    }

    final Set<String> docTerms = new ObjectOpenHashSet<>();
    Terms terms = statsProvider.getTermVector(doc, docField);

    if (terms == null) {
      return docTerms;
    }

    try {
      TermsEnum termsEnum = terms.iterator();
      BytesRef text;

      while ((text = termsEnum.next()) != null) {
        String term = text.utf8ToString();
        docTerms.add(term);
      }

    } catch (final IOException e) {
      throw new RuntimeException(e);
    }

    cacheTerms.put(doc, docTerms);
    return docTerms;
  }

  @Override
  public double computeBackgroundProb(String term) {
    if (cacheBackground.containsKey(term)) {
      return cacheBackground.get(term);
    }
    final long termTotalFrequency = statsProvider.getTotalTermFrequency(term, docField);
    final long collectionLength = statsProvider.getCollectionTokensSize(docField);
    double value = (double) termTotalFrequency / collectionLength;
    cacheBackground.put(term, value);
    return value;
  }

  @Override
  public boolean termExists(String term) {
    long ttf = statsProvider.getTotalTermFrequency(term, docField);
    return ttf > 0;
  }
}