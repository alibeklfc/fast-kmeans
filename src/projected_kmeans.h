#ifndef PROJECTED_KMEANS_H
#define PROJECTED_KMEANS_H

/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include "original_space_kmeans.h"
#include "dataset.h"
#include "hamerly_kmeans.h"

class ProjectedKmeans : public OriginalSpaceKmeans{
public:
    virtual ~ProjectedKmeans() { free(); }
    virtual std::string getName() const { return "projected"; }
    void setReducedDim(double d) { redDim = d; }


private:
    int redDim;

    Dataset* reduced;
    Dataset* redData;
    Dataset* outCenters;


    HamerlyKmeans* algorithm;
    HamerlyKmeans* algorithm2;

    void reduceDataset(Dataset* x);
    virtual int runThread(int threadId, int maxIterations);

    void GramSchmidt(const Dataset *x, unsigned short k, Dataset *reduced, int redDim);

    void normalizeMatrix(Dataset *reduced, int redDim, const Dataset *x);

    void multiplyMatrix(Dataset *reduced, int redDim, const Dataset *x, Dataset *redData);

    void generateRandomMatrix(int redDim, const Dataset *x, Dataset *reduced);
};
#endif

