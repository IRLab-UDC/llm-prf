package org.irlab.ecir26.searcher.rf;

import org.irlab.ecir26.searcher.smoothing.Smoothing;
import org.irlab.ecir26.searcher.util.TermWeights;

import java.util.Map;

public abstract class AbstractRelevanceFeedback implements RelevanceFeedback {

  protected final String docField;
  protected final Smoothing documentSmoothing;

  public AbstractRelevanceFeedback(String docField, Smoothing documentSmoothing) {

    this.docField = docField;
    this.documentSmoothing = documentSmoothing;
  }

  protected abstract TermWeights estimateWeights(Map<Integer, Double> relevanceSet);

  protected abstract String getName();

  @Override
  public String toString() {
    return getName();
  }
}