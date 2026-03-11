package org.irlab.llmprf.searcher.rf;

import org.irlab.llmprf.searcher.smoothing.Smoothing;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public abstract class AbstractRelevanceFeedback implements RelevanceFeedback {

  protected final String docField;
  protected final Smoothing smoothing;

  public AbstractRelevanceFeedback(String docField, Smoothing smoothing) {
    this.docField = docField;
    this.smoothing = smoothing;
  }

  protected Set<String> getVocab(Map<Integer, Double> relevanceSet) {
    final Set<String> vocab = new HashSet<>();
    relevanceSet.forEach((doc, ql) -> {
      final Set<String> docTerms = smoothing.getDocTerms(doc);
      vocab.addAll(docTerms);
    });
    return vocab;
  }
}