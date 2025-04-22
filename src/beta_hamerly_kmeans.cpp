/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include "beta_hamerly_kmeans.h"
#include "general_functions.h"
#include <cmath>
#include <algorithm>
#include <vector>

int BetaHamerlyKmeans::runThread(int threadId, int maxIterations) {
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

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;
        int currBin = -1;
        int count = 0;
        // compute the inter-center distances, keeping only the closest distances
        update_s(threadId);
        synchronizeAllThreads();

        // loop over all records
        for (int i = startNdx; i < endNdx; ++i) {

            unsigned short closest = assignment[normIndexPairs[i].second];

            // if upper[i] is less than the greater of these two, then we can
            // ignore record i
            double upper_comparison_bound = std::max(s[closest], lower[normIndexPairs[i].second]);

            // first check: if u(x) <= s(c(x)) or u(x) <= lower(x), then ignore
            // x, because its closest center must still be closest
            if (upper[normIndexPairs[i].second] <= upper_comparison_bound) {
                continue;
            }

            // otherwise, compute the real distance between this record and its
            // closest center, and update upper
            double u2 = pointCenterDist2(normIndexPairs[i].second, closest);
            upper[normIndexPairs[i].second] = sqrt(u2);

            // if (u(x) <= s(c(x))) or (u(x) <= lower(x)), then ignore x
            if (upper[normIndexPairs[i].second] <= upper_comparison_bound) {
                continue;
            }

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

            // now update the lower bound by looking at all other centers
            double l2 = std::numeric_limits<double>::max(); // the squared lower bound
            for (int j = 0; j < count; ++j) {
                if (listOfEligibleCenters[j] == closest) { continue; }

                double dist2 = pointCenterDist2(normIndexPairs[i].second, listOfEligibleCenters[j]);

                if (dist2 < u2) {
                    // another center is closer than the current assignment

                    // change the lower bound to be the current upper bound
                    // (since the current upper bound is the distance to the
                    // now-second-closest known center)
                    l2 = u2;

                    // adjust the upper bound and the current assignment
                    u2 = dist2;
                    closest = listOfEligibleCenters[j];
                } else if (dist2 < l2) {
                    // we must reduce the lower bound on the distance to the
                    // *second* closest center to x[i]
                    l2 = dist2;
                }
            }
            for (int j = 0; j < k; ++j) {
                if (closest == j) {
                    alpha[currBin * k + j]++;
                } else {
                    beta[currBin * k + j]++;
                }
            }
            // we have been dealing in squared distances; need to convert
            lower[normIndexPairs[i].second] = sqrt(l2);

            // if the assignment for i has changed, then adjust the counts and
            // locations of each center's accumulated mass
            if (assignment[normIndexPairs[i].second] != closest) {
                upper[normIndexPairs[i].second] = sqrt(u2);
                changeAssignment(normIndexPairs[i].second, closest, threadId);
            }
        }
        for (int i = 0; i < k * numberOfBins; i++){
            probabilities[i] = (double)(alpha[i] - 1) / (alpha[i] + beta[i] - 2);
        }
        verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6
        // calculate the new center locations
        synchronizeAllThreads();
        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        synchronizeAllThreads();

        if (! converged) {
            update_bounds(startNdx, endNdx);
        }

        synchronizeAllThreads();
    }

    return iterations;
}


/* This method does the following:
 *  - finds the furthest-moving center
 *  - finds the distances moved by the two furthest-moving centers
 *  - updates the upper/lower bounds for each record
 *
 * Parameters:
 *  - startNdx: the first index of the dataset this thread is responsible for
 *  - endNdx: one past the last index of the dataset this thread is responsible for
 */
void BetaHamerlyKmeans::update_bounds(int startNdx, int endNdx) {
    double longest = centerMovement[0], secondLongest = (1 < k) ? centerMovement[1] : centerMovement[0];
    int furthestMovingCenter = 0;

    if (longest < secondLongest) {
        furthestMovingCenter = 1;
        std::swap(longest, secondLongest);
    }

    for (int j = 2; j < k; ++j) {
        if (longest < centerMovement[j]) {
            secondLongest = longest;
            longest = centerMovement[j];
            furthestMovingCenter = j;
        } else if (secondLongest < centerMovement[j]) {
            secondLongest = centerMovement[j];
        }
    }

    // update upper/lower bounds
    for (int i = startNdx; i < endNdx; ++i) {
        // the upper bound increases by the amount that its center moved
        upper[i] += centerMovement[assignment[i]];

        // The lower bound decreases by the maximum amount that any center
        // moved, unless the furthest-moving center is the one it's assigned
        // to. In the latter case, the lower bound decreases by the amount
        // of the second-furthest-moving center.
        lower[i] -= (assignment[i] == furthestMovingCenter) ? secondLongest : longest;
    }
}