package algorithm_dbscan

import "math"

type DBSCAN struct {
	data         [][]float64
	visited      []int64
	eps          float64
	minSamples   int
}

type outData struct {
	Dimension     int64
	CountClusters int64
	Predict       []int64
}

func (k *DBSCAN) InitAlgorithm(minSamples int, eps float64) {
	k.minSamples = minSamples
	k.eps = eps
}

//Fit accepts a data set and divides it into clusters using the DBSCAN method
func (k *DBSCAN) Fit(data [][]float64) *outData {
	k.data = data
	k.visited = make([]int64, len(k.data))
	var indCluster int64 = 0
	for indPoint := range k.data {
		if k.visited[indPoint] == 0 {
			indCluster++
			neighbours := k.findNeighbours(indPoint)
			if len(neighbours) >= k.minSamples {
				k.visited[indPoint] = indCluster
				k.expandCluster(indCluster, neighbours)
			} else {
				k.visited[indPoint] = -1
			}
		}
	}
	return &outData{
		Dimension:     int64(len(k.data[0])),
		CountClusters: indCluster,
		Predict:       k.visited,
	}
}

//expandCluster expands the cluster and find neighbors neighbors
func (k *DBSCAN) expandCluster(indCluster int64, neighbours []int) {
	// neighbours contain neighbor indexes
	for _, indPoint := range neighbours {
		if k.visited[indPoint] == -1 {
			k.visited[indPoint] = indCluster
		}
		if k.visited[indPoint] == 0 {
			k.visited[indPoint] = indCluster
			allNeighbours := k.findNeighbours(indPoint)
			if len(allNeighbours) >= k.minSamples{
				neighbours = append(neighbours, allNeighbours...)
			}
		}
	}
}

// findNeighbours finds neighbors in epsilon neighborhood
func (k *DBSCAN) findNeighbours(ind_point_1 int) []int {
	neighbours := make([]int, 0)
	point1 := k.data[ind_point_1]
	for indPoint2, point2 := range k.data {
		if indPoint2 != ind_point_1 {
			if euclideanDistance(point1, point2) <= k.eps {
				neighbours = append(neighbours, indPoint2)
			}
		}
	}
	return neighbours
}

// euclideanDistance calculates the distance between points
func euclideanDistance(point1 []float64, point2 []float64) float64 {
	dist := 0.0
	for i := range point1 {
		dist += (point1[i] - point2[i]) * (point1[i] - point2[i])
	}
	return math.Sqrt(dist)
}

//Accuracy calculates the quality metric
func (k *DBSCAN) Accuracy(predict []int, labels []int) float64 {
	metrica := 0
	for i := range predict {
		if labels[i] == predict[i] {
			metrica++
		}
	}
	return float64(metrica) / float64(len(labels))
}

//func (k *DBSCAN) Predict() {
//
//}
