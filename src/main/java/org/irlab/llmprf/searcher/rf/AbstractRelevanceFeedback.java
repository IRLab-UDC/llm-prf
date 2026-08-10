package org.irlab.llmprf.searcher.rf;

import org.irlab.llmprf.searcher.smoothing.Smoothing;

import java.util.HashMap;
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
    relevanceSet.forEach((doc, ql) -> vocab.addAll(smoothing.getDocTerms(doc)));
    return vocab;
  }

  protected Set<String> getVocab(Map<Integer, Double> relevanceSet,
                                  Map<Integer, Set<String>> allowedTerms) {
    if (allowedTerms == null || allowedTerms.isEmpty()) {
      return getVocab(relevanceSet);
    }
    final Set<String> vocab = new HashSet<>();
    relevanceSet.forEach((doc, ql) -> {
      Set<String> docTerms = new HashSet<>(smoothing.getDocTerms(doc));
      Set<String> allowed = allowedTerms.get(doc);
      if (allowed != null && !allowed.isEmpty()) {
        docTerms.retainAll(allowed);
      }
      vocab.addAll(docTerms);
    });
    return vocab;
  }

  protected Set<String> getVocabFromSpans(Map<Integer, Double> relevanceSet,
                                           Map<Integer, Map<String, Integer>> spanTermFreqs) {
    final Set<String> vocab = new HashSet<>();
    relevanceSet.forEach((doc, ql) -> {
      Map<String, Integer> spanTF = spanTermFreqs.get(doc);
      if (spanTF != null) {
        spanTF.keySet().stream().filter(smoothing::termExists).forEach(vocab::add);
      } else {
        vocab.addAll(smoothing.getDocTerms(doc));
      }
    });
    return vocab;
  }

  protected Map<Integer, Long> spanLengths(Map<Integer, Map<String, Integer>> spanTermFreqs) {
    Map<Integer, Long> lengths = new HashMap<>();
    spanTermFreqs.forEach((doc, tf) ->
        lengths.put(doc, tf.values().stream().mapToLong(Integer::longValue).sum()));
    return lengths;
  }
}
