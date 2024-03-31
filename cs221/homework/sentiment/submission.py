#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    result={}
    for word in x.split(" "):
        if word in result:
            result[word] = result[word] + 1
        else:
            result[word] = 1
    return result
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    def score(w, feature):
        return dotProduct(w, feature);
    def margin(w, feature, y):
        return score(w,feature) * y
    def loss_hinge(w,feature,y):
        return max(1 - margin(w,feature,y), 0);
    def gradient_loss_hinge(w, feature, y):
        gradient={};
        for word in feature:
            gradient[word] = feature[word] * -y;
        return gradient;
    def sF(w,feature,y):
        return loss_hinge(w,feature,y);
    def sdF(w,feature,y):
        return gradient_loss_hinge(w,feature,y);

    for x,y in trainExamples:
        for word in featureExtractor(x):
            weights[word] = 0.0;

    for i in range(0, numEpochs):
        for x, y in trainExamples:
            feature = featureExtractor(x);
            value = sF(weights, feature, y)
            if (value >0):
                gradient = sdF(weights, feature, y)
                increment(weights, -eta, gradient);
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        y = None
        for key in weights:
            value = random.randint(1,100);
            phi[key] = value;
        y = 1 if dotProduct(phi, weights) >= 0 else -1;
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        result = {};
        input = x.replace(" ", "");
        if (n >= len(input)):
            return {x: 1};
        start = 0;
        while(start + n <= len(input)):
            substr = input[start:(start+n)]
            if substr in result:
                result[substr] = result[substr] + 1;
            else:
                result[substr] = 1;
            start +=1;
        return result;
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)

    # squared distance between the point (sparse array) and the centroid (sparse array):
    def squared_dis(pre_sum: float, point: Dict[str, float],
                    centroid_sum: float, centroid: Dict[str, float]):
        dis = pre_sum + centroid_sum;
        for key in point.keys():
            dis += -2 * point.get(key,0) * centroid.get(key,0);
        return dis;
    def getAssignments(point_square_sum: List[float],
                       points: List[Dict[str, float]], centroids_sum: List[float],
                       centroids: List[Dict[str, float]]):
#        print('assignment');
        result = [];
        for point_index in range(0, len(points)):
            min_dis = float('inf');
            point = points[point_index];
            assignment = 0;
            pre_sum = point_square_sum[point_index];
            for i in range(0, len(centroids)):
                centroid  = centroids[i];
#                key = (point, centroid);
#                if key in cache:
#                    dis = cache[key];
#                else:
                dis = squared_dis(pre_sum, point, centroids_sum[i], centroid);
#                    cache[key] = dis;
#                print('dis:{}, min_dis:{}'.format(dis, min_dis));
                if min_dis > dis:
                    assignment = i;
                    min_dis = dis;
#            print("{} -> {}".format(point_index, assignment));
            result.append(assignment);
#        print("point:{}, \nassgiment:{} \ncentroids:{}".format(points, result, centroids));
        return result;

    def setCentroids(points: List[Dict[str, float]], centroids: List[Dict[str, float]],
                     assignments: List[int]):
#        print('centroids');
        clusters = [[] for _ in range(0, len(centroids))];
        for i in range(0, len(assignments)):
            assignment = assignments[i];
            point = points[i];
            cluster = clusters[assignment];
            cluster.append(point);
#            print("point:{}, assgiment:{} cluster:{}".format(point, assignment, cluster));
#        print("clusters:{}".format(clusters));
        for i in range(0, len(clusters)):
            cluster = clusters[i];
            if len(cluster) == 0:
                continue;
            new_centroid = {};
            scale = 1.0 / len(cluster);
            for point in cluster:
                increment(new_centroid, scale, point);
            centroids[i] = new_centroid;

    def get_squared_sum(points: List[Dict[str, float]]):
        point_square_sum = [];
        for point in points:
            point_square_sum.append( sum([ v **2 for v in point.values()]));
        return point_square_sum;

    def impl():
        centroids = [ examples[random.randint(0, len(examples)-1)] for i in range(0,K)]
        assignments= []
        loss = 0.0;
        point_square_sum = get_squared_sum(examples);
#        print(examples);
#        print(point_square_sum);
        for _ in range(0, maxEpochs):
#            print("epoch:{}".format(_));
            centroids_sum = get_squared_sum(centroids);
            new_assignments = getAssignments(point_square_sum, examples, centroids_sum, centroids);
            if assignments == new_assignments:
                break;
            assignments = new_assignments;
            setCentroids(examples, centroids, assignments);

        centroids_sum = get_squared_sum(centroids);
        for i in range(0, len(assignments)):
            assignment = assignments[i];
            point = examples[i];
            centroid = centroids[assignment];
            loss += squared_dis(point_square_sum[i], point, centroids_sum[assignment],centroid);
        return (centroids, assignments, loss);

    return impl();
    # END_YOUR_CODE
