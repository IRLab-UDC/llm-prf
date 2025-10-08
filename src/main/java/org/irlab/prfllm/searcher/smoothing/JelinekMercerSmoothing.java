package org.irlab.prfllm.searcher.smoothing;

import org.irlab.prfllm.searcher.util.StatsProvider;

public final class JelinekMercerSmoothing extends AbstractSmoothing {

    public JelinekMercerSmoothing(double smoothingParameter, String docField, StatsProvider statsProvider) {

        super(smoothingParameter, docField, statsProvider);
    }

    public JelinekMercerSmoothing(double smoothingParameter, String docField, StatsProvider statsProvider, StatsProvider statsProviderBackground) {

        super(smoothingParameter, docField, statsProvider, statsProviderBackground);
    }

    @Override
    public double computeValue(String term, int doc) {

        return computeMLE(term, doc) * (1 - smoothingParameter) + smoothingParameter * computeBackgroundProb(term);
    }

    @Override
    public String getName() {

        return String.format("JM-lambda-%1.2f", smoothingParameter);
    }
}