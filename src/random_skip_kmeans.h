#ifndef RANDOM_SKIP_KMEANS_H
#define RANDOM_SKIP_KMEANS_H

/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include "original_space_kmeans.h"

class RandomSkipKmeans : public OriginalSpaceKmeans {
    public:
        RandomSkipKmeans(double perc) : percentage(perc) {}
        virtual std::string getName() const { return "randomskip"; }
        virtual ~RandomSkipKmeans() { free(); }

protected:
        virtual int runThread(int threadId, int maxIterations);
        double percentage;
};

#endif

