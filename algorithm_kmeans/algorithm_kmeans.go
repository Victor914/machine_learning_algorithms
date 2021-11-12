package algorithm_kmeans

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"
)

type KMeans struct {
	data         [][]float64
	clusters     int
	dimension    int
	centers      [][]float64
	membership   []int
	maxIter      int
	acc          []float64
	countsPoints []int
}

// InitAlgorithm defines the parameters.
func (k *KMeans) InitAlgorithm(clusters, dimension, maxIter int) {
	k.clusters = clusters
	k.dimension = dimension
	k.maxIter = maxIter
}

type outFit struct {
	Clusters     int
	Dimension    int
	Centers      [][]float64
	Predict      []int
	CountIters   int
	CountsPoints []int
	Metrics      []float64
	Accuracy     []float64
}

// Fit enters data and initializes the initial centers.
func (k *KMeans) Fit(data [][]float64, labels []int, iter_accur, auto_centers bool) (error, *outFit) {
	k.data = data
	if auto_centers {
		k.InitCenters()
	} else if len(k.centers) == 0 {
		return errors.New("initial centers are not defined"), nil
	}

	k.membership = make([]int, len(k.data))
	metric_1 := k.distributionPoints()
	k.changeCenters()
	metric_2 := k.distributionPoints()
	k.changeCenters()
	iter, m := 2, []float64{}
	m = append(m, metric_1)
	m = append(m, metric_2)
	for 0 != math.Abs(metric_2-metric_1) && iter < k.maxIter {
		if iter_accur {
			k.acc = append(k.acc, k.Accuracy(k.membership, labels))
		}
		metric_1, metric_2 = metric_2, k.distributionPoints()
		m = append(m, metric_2)
		k.changeCenters()
		iter++
	}
	return nil, &outFit{
		Clusters:     k.clusters,
		Dimension:    k.dimension,
		Centers:      k.centers,
		Predict:      k.membership,
		CountsPoints: k.countsPoints,
		CountIters:   iter,
		Metrics:      m,
		Accuracy:     k.acc,
	}
}

func (k *KMeans) maxCoord() []float64 {
	m := make([]float64, len(k.data[0]))
	for indCoord := range k.data[0] {
		for indPoint := range k.data {
			if k.data[indPoint][indCoord] > m[indCoord] || indPoint == 0 {
				m[indCoord] = k.data[indPoint][indCoord]
			}
		}
	}
	return m
}

func (k *KMeans) minCoord() []float64 {
	m := make([]float64, len(k.data[0]))
	for indCoord := range k.data[0] {
		for indPoint := range k.data {
			if k.data[indPoint][indCoord] < m[indCoord] || indPoint == 0 {
				m[indCoord] = k.data[indPoint][indCoord]
			}
		}
	}
	return m
}

type outPredict struct {
	Clusters     int
	Dimension    int
	Centers      [][]float64
	Predict      []int
	CountsPoints []int
	Accuracy     float64
}

// Predict performs clustering.
func (k *KMeans) Predict(data [][]float64) *outPredict {
	memb := make([]int, len(data))
	for indPoint := range data {
		minDist := 0.0
		for indCenter := range k.centers {
			distance := euclideanDistance(data[indPoint], k.centers[indCenter])
			if distance < minDist || indCenter == 0 {
				minDist = distance
				memb[indPoint] = indCenter
			}
		}
	}
	return &outPredict{
		Clusters:     k.clusters,
		Dimension:    k.dimension,
		Centers:      k.centers,
		Predict:      memb,
		CountsPoints: k.countsPoints,
	}
}

//Accuracy calculates the quality metric
func (k *KMeans) Accuracy(membership []int, labels []int) float64 {
	metrica := 0
	for i := range membership {
		if labels[i] == membership[i] {
			metrica++
		}
	}
	return float64(metrica) / float64(len(labels))
}

// GetCenters prints and returns the set centers
func (k *KMeans) GetCenters() [][]float64 {
	fmt.Println(k.centers)
	return k.centers
}

// SetCenters installs users centers.
func (k *KMeans) SetCenters(centers [][]float64) {
	k.centers = centers
}

// euclideanDistance
func euclideanDistance(point1 []float64, point2 []float64) float64 {
	dist := 0.0
	for i := range point1 {
		dist += (point1[i] - point2[i]) * (point1[i] - point2[i])
	}
	return math.Sqrt(dist)
}

// distributionPoints distributes points into clusters.
func (k *KMeans) distributionPoints() float64 {
	totalDistance := 0.0
	for indPoint := range k.data {
		minDist := 0.0
		for IndCenter := range k.centers {
			distance := euclideanDistance(k.data[indPoint], k.centers[IndCenter])
			if distance < minDist || IndCenter == 0 {
				minDist = distance
				k.membership[indPoint] = IndCenter
			}
		}
	}
	for i := range k.membership {
		totalDistance += euclideanDistance(k.data[i], k.centers[k.membership[i]])
	}
	return totalDistance
}

// changeCenters recalculates the coordinates of the cluster centers.
// Performs the M step.
func (k *KMeans) changeCenters() {
	k.countsPoints = make([]int, len(k.centers))
	for indCenter := range k.centers {
		count := 0.0
		sumsCoord := make([]float64, k.dimension)
		for indPoint := range k.data {
			if indCenter == k.membership[indPoint] {
				count++
				for indCoord := range sumsCoord {
					sumsCoord[indCoord] += k.data[indPoint][indCoord]
				}
			}
		}
		k.countsPoints[indCenter] = int(count)
		if count == 0 {
			min, max := k.minCoord(), k.maxCoord()
			rand.Seed(time.Now().UTC().Unix())
			for indCoord := range k.centers[0] {
				k.centers[indCenter][indCoord] = (max[indCoord]-min[indCoord])*rand.Float64() + min[indCoord]
			}
		} else {
			for i := range sumsCoord {
				k.centers[indCenter][i] = sumsCoord[i] / count
			}
		}
	}
}

// InitCenters initializes the initial cluster centers. It divides all the data
// by the number of classes and assigns the first coordinates of the pieces to
//  the corresponding class centers.
func (k *KMeans) InitCenters() {
	k.centers = make([][]float64, k.clusters)
	min, max := k.minCoord(), k.maxCoord()
	for indCenter := range k.centers {
		k.centers[indCenter] = make([]float64, k.dimension)
		for indCoord := range k.centers[indCenter] {
			rand.Seed(time.Now().UTC().Unix())
			k.centers[indCenter][indCoord] = (max[indCoord]-min[indCoord])*rand.Float64() + min[indCoord]
			//k.data[indCenter*len(k.data)/k.clusters][indCoord]
			//k.data[rand.Intn(len(k.data))][rand.Intn(len(k.data[0]))]
		}
	}
}
