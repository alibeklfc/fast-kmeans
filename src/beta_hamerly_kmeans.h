#ifndef BETA_HAMERLY_KMEANS_H
#define BETA_HAMERLY_KMEANS_H

/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include "triangle_inequality_base_kmeans.h"

class BetaHamerlyKmeans : public TriangleInequalityBaseKmeans {
public:
    BetaHamerlyKmeans() { numLowerBounds = 1; }
    virtual ~BetaHamerlyKmeans() { free(); }
    virtual std::string getName() const { return "betahamerly"; }

protected:
    // Update the upper and lower bounds for the given range of points.
    double *pointNorms;
    int *beta;
    int *alpha;
    double *probabilities;

    void update_bounds(int startNdx, int endNdx);

    virtual int runThread(int threadId, int maxIterations);
};

#endif