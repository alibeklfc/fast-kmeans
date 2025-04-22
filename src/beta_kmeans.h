#ifndef BETA_KMEANS_H
#define BETA_KMEANS_H

/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include <unordered_map>
#include <vector>
#include "original_space_kmeans.h"

class BetaKmeans : public OriginalSpaceKmeans {
    public:
        virtual std::string getName() const { return "beta"; }
        virtual ~BetaKmeans() { free(); }
    protected:
        virtual int runThread(int threadId, int maxIterations);
        double *pointNorms;
        int *beta;
        int *alpha;
        double *probabilities;
};

#endif

