/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include "random_skip_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cstring>
#include <random>


/* The classic algorithm of assign, move, repeat. No optimizations that prune
 * the search.
 *
 * Return value: the number of iterations performed (always at least 1)
 */

int RandomSkipKmeans::runThread(int threadId, int maxIterations) {
    std::random_device rd;
    std::mt19937 gen(rd());  // Seeded with a static value


    // track the number of iterations the algorithm performs
    int iterations = 0;


    int startNdx = start(threadId);
    int endNdx = end(threadId);

    std::vector<int> indices(endNdx - startNdx);
    std::iota(indices.begin(), indices.end(), startNdx);

    int numToProcess = static_cast<int>((1.0 - this->percentage) * indices.size());

    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<int>::max());
    while ((iterations < maxIterations) && (! converged)) {
        ++iterations;
        for (int i = 0; i < numToProcess; ++i) {
            int j = i + dist(gen) % (n - i);
            std::swap(indices[i], indices[j]);
        }

        // loop over all examples
        for (int t = 0; t < numToProcess; ++t) {
            int i = indices[t];
            // look for the closest center to this example
            int closest = assignment[i];
            double closestDist2 = std::numeric_limits<double>::max();
            for (int j = 0; j < k; ++j) {
                double d2 = pointCenterDist2(i, j);
                #ifdef COUNT_DISTANCES
                numDistances += 1;
                #endif
                if (d2 < closestDist2) {
                    closest = j;
                    closestDist2 = d2;
                }
            }
            if (assignment[i] != closest) {
                changeAssignment(i, closest, threadId);
            }
        }

        verifyAssignment(iterations, startNdx, endNdx);

        synchronizeAllThreads();

        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        synchronizeAllThreads();
    }

    return iterations;
}

