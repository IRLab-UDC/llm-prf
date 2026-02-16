package org.irlab.ecir26.searcher.util.scorers;

/**
 * Common result class for LLM-based relevance judgments.
 * Used by both MonoT5 and VLLM scorers to provide a uniform interface.
 */
public class LLMResult {
  public final boolean isRelevant;
  public final double probTrue;
  public final double score;

  /**
   * Create an LLM result.
   *
   * @param isRelevant Binary relevance judgment (true/false)
   * @param probTrue   Probability of relevance (0.0 to 1.0)
   * @param score      Overall relevance score (can be logit, probability, or other)
   */
  public LLMResult(boolean isRelevant, double probTrue, double score) {
    this.isRelevant = isRelevant;
    this.probTrue = probTrue;
    this.score = score;
  }

  @Override
  public String toString() {
    return String.format("LLMResult(isRelevant=%s, probTrue=%.4f, score=%.4f)", isRelevant, probTrue, score);
  }
}
