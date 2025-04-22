/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include "beta_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <algorithm>



int BetaKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;
    int startNdx = start(threadId);
    int endNdx = end(threadId);
    int* listOfEligibleCenters = new int[k];

    pointNorms = new double[n];
    for (int i = startNdx; i < endNdx; ++i) {
        double sumOfSquares = 0.0;
        for (int dim = 0; dim < d; ++dim) {
            double coord = x->data[i * d + dim];
            sumOfSquares += coord * coord;
        }
        pointNorms[i] = sqrt(sumOfSquares);
    }

    std::vector<std::pair<double, int>> normIndexPairs(n);
    for (int i = 0; i < n; ++i) {
        normIndexPairs[i] = {pointNorms[i], i};
    }

    std::sort(normIndexPairs.begin(), normIndexPairs.end());

    const int numberOfBins = 11;
    const int pointsPerBin = n / numberOfBins;
    const double threshold = 1.0 / pointsPerBin;

    beta = new int[k * numberOfBins];
    alpha = new int[k * numberOfBins];
    probabilities = new double[k * numberOfBins];

    std::fill(beta, beta + k * numberOfBins, 2.0);
    std::fill(alpha, alpha + k * numberOfBins, 2.0);
    std::fill(probabilities, probabilities + k * numberOfBins, 0.5);

    while ((iterations < maxIterations) && (!converged)) {
        ++iterations;
        int currBin = -1;
        int count = 0;

        for (int i = startNdx; i < endNdx; ++i) {
            int getBin = i / pointsPerBin;
            if (getBin == numberOfBins) {
                getBin = numberOfBins - 1;
            }
            if (getBin != currBin) {
                count = 0;
                for (int j = 0; j < k; ++j) {
                    bool skip = (getBin == 0 && probabilities[getBin * k + j] < threshold && probabilities[(getBin + 1) * k + j] < threshold) ||
                                (getBin == numberOfBins - 1 && probabilities[getBin * k + j] < threshold && probabilities[(getBin - 1) * k + j] < threshold) ||
                                (getBin > 0 && getBin < numberOfBins - 1 && probabilities[getBin * k + j] < threshold && probabilities[(getBin - 1) * k + j] < threshold && probabilities[(getBin + 1) * k + j] < threshold);
                    if (!skip) {
                        listOfEligibleCenters[count++] = j;
                    }
                }
                currBin = getBin;
            }

            int closest = assignment[normIndexPairs[i].second];
            double closestDist2 = sqrt(pointCenterDist2(normIndexPairs[i].second,
                                                                 assignment[normIndexPairs[i].second]));

            #ifdef COUNT_DISTANCES
            numDistances += 1;
            #endif
            for (int j = 0; j < count; ++j) {
                if (listOfEligibleCenters[j] == assignment[normIndexPairs[i].second]) {
                    continue;
                }
                #ifdef COUNT_DISTANCES
                numDistances += 1;
                #endif
                double d2 = std::sqrt(pointCenterDist2(normIndexPairs[i].second,
                                                           listOfEligibleCenters[j]));
                if (d2 < closestDist2) {
                    closest = listOfEligibleCenters[j];
                    closestDist2 = d2;
                }
            }

            for (int j = 0; j < k; ++j) {
                if (closest == j) {
                    alpha[currBin * k + j]++;
                } else {
                    beta[currBin * k + j]++;
                }
            }

            if (assignment[normIndexPairs[i].second] != closest) {
                changeAssignment(normIndexPairs[i].second, closest, threadId);
            }
        }

        for (int i = 0; i < k * numberOfBins; ++i) {
            probabilities[i] = (double)(alpha[i] - 1) / (alpha[i] + beta[i] - 2);
        }

        verifyAssignment(iterations, startNdx, endNdx);
        synchronizeAllThreads();

        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        synchronizeAllThreads();
    }

    delete[] listOfEligibleCenters;
    return iterations;
}
