/* Authors: Greg Hamerly and Alibek Zhakubayev
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2025
 */

#include "naive_kmeans.h"
#include "general_functions.h"
#include "projected_kmeans.h"
#include <cassert>
#include <cstring>
#include <random>

/* The classic algorithm of assign, move, repeat. No optimizations that prune
 * the search.
 *
 * Return value: the number of iterations performed (always at least 1)
 */
double getDistortion2(Dataset const *x, unsigned short *assignment, Dataset *centers){
    double result = 0.0;
    for(int i = 0; i < x->n; i++){
        double temp = 0;
        for(int j = 0; j < x->d; j++){
            temp = temp + (x->data[i * x->d + j] - centers->data[assignment[i] * x->d + j]) *
                          (x->data[i * x->d + j] - centers->data[assignment[i] * x->d + j]);
        }
        result = result + temp;
    }
    return result / x->n;
}

int ProjectedKmeans::runThread(int threadId, int maxIterations) {

    delete reduced;
    reduced = new Dataset(x->d, this->redDim);
    generateRandomMatrix(this->redDim, x, reduced);
    GramSchmidt(x, k, reduced, this->redDim);
    normalizeMatrix(reduced, this->redDim, x);

    delete redData;
    redData = new Dataset(x->n, this->redDim);
    multiplyMatrix(reduced, this->redDim, x, redData);

    algorithm = new HamerlyKmeans();
    algorithm2 = new HamerlyKmeans();

    algorithm->initialize(redData, k, assignment, numThreads);
    algorithm->run(maxIterations);

    delete outCenters;
    outCenters = new Dataset(k, x->d);

    algorithm2->initialize(x, k, assignment, numThreads);
    int finalIterations = algorithm2->run(40);

    *outCenters = *algorithm2->getCenters();

    this->centers = outCenters;

    return finalIterations;
}

void ProjectedKmeans::GramSchmidt(Dataset const *x, unsigned short k, Dataset *reduced, int redDim){
    double *temp = new double[x->d];
    for (int i = 0; i < redDim; ++i) {
        //Set temp to zero
        for (int k = 0; k < x->d; k++){
            temp[k] = 0;
        }
        //Transformed vector Vi = Initial vector Wi - <Wi, Vj (j < i)> / <Vj, Vj>
        for (int j = 0; j < i; ++j) {
            //Numtemp and dentemp are the numerator and denominator of one subtraction
            double numtemp = 0;
            double dentemp = 0;
            for (int k = 0; k < x->d; k++){
                numtemp = numtemp + reduced->data[k * redDim + i] * reduced->data[k * redDim + j];
                dentemp = dentemp + reduced->data[k * redDim + j] * reduced->data[k * redDim + j];
            }
            //temp is the sum of all vectors we need to subtract
            for (int k = 0; k < x->d; k++){
                temp[k] = temp[k] + (numtemp / dentemp) * reduced->data[k * redDim + j];
            }
        }
        //Subtract temp from reduced
        for (int k = 0; k < x->d; k++){
            reduced->data[k * redDim + i] = reduced->data[k * redDim + i] - temp[k];
        }
    }
    delete [] temp;

}

void ProjectedKmeans::normalizeMatrix(Dataset *reduced, int redDim, Dataset const *x){
    for (int i = 0; i < redDim; ++i) {
        double total = 0;
        for (int k = 0; k < x->d; k++){
            total = total + reduced->data[k * redDim + i] * reduced->data[k * redDim + i];
        }
        total = sqrt(total);
        for (int k = 0; k < x->d; k++){
            reduced->data[k * redDim + i] = reduced->data[k * redDim + i]  / total;
        }
    }
}
void ProjectedKmeans::multiplyMatrix(Dataset *reduced, int redDim, Dataset const *x, Dataset *redData){
    for(int i = 0; i < x->n; i++){
        for(int j = 0; j < redDim; j++){
            redData->data[redDim * i + j] = 0;
            for(int k = 0; k < x->d; k++){
                redData->data[redDim * i + j] = redData->data[redDim * i + j] + x->data[i * x->d + k] * reduced->data[k * redDim + j];
            }
        }
    }
}

void ProjectedKmeans::generateRandomMatrix(int redDim, Dataset const *x, Dataset *reduced){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,1);

    // Fill the reduced matrix
    for (int i = 0; i < x->d * redDim; ++i) {
        reduced->data[i] = distribution(generator);
    }
};