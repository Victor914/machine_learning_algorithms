package algorithm_kmeans

import (
	"fmt"
	"github.com/cdipaolo/goml/cluster"
	"image/color"
	"machine_learning_algorithm/algorithm_kmeans"
	"time"

	//"github.com/cdipaolo/goml/cluster"
	//"gonum.org/v1/plot/vg/draw"
	//"image/color"
	"io/ioutil"
	"machine_learning_algorithm/make_plot"

	//"machine_learning_algorithm/algorithm_kmeans"
	//"machine_learning_algorithm/make_plot"
	"math/rand"
	"strconv"
	"strings"
)
go
func main() {

	////link := "./data.txt"
	////data, err := readAndPrepareDate(link, dimension)
	////if nil != err {
	////	log.Fatalf("could not read %v: %v", link, err)
	////}
	//// Generating input data.
	//var gaussian [][]float64
	//for i := 0; i < 40; i++ {
	//	x := rand.NormFloat64() + 4
	//	y := rand.NormFloat64()*0.25 + 5
	//	gaussian = append(gaussian, []float64{x, y})
	//}
	//for i := 0; i < 66; i++ {
	//	x := rand.NormFloat64()
	//	y := rand.NormFloat64() + 10
	//	gaussian = append(gaussian, []float64{x, y})
	//}
	//for i := 0; i < 100; i++ {
	//	x := rand.NormFloat64()*3 - 10
	//	y := rand.NormFloat64()*0.25 - 7
	//	gaussian = append(gaussian, []float64{x, y})
	//}
	//for i := 0; i < 23; i++ {
	//	x := rand.NormFloat64() * 2
	//	y := rand.NormFloat64() - 1.25
	//	gaussian = append(gaussian, []float64{x, y})
	//}
	//plot := new(make_plot.PltStruct)
	//plot.CreateScatter(
	//	"1.Data position.png",
	//	gaussian,
	//	draw.CircleGlyph{},
	//	color.RGBA{129,0,0,185},
	//	5)
	//
	////Define the initial vector
	//centers := make([][]float64, 4)
	//centers[0] = []float64{-15, -6}
	//centers[1] = []float64{5, -5}
	//centers[2] = []float64{6, 6}
	//centers[3] = []float64{-5, 7}
	//
	//alg := new(algorithm_kmeans.KMeans)
	//alg.InitAlgorithm(clusters, dimension, 1000)
	//alg.Fit(gaussian)
	//alg.SetCenters(centers)
	//plot.AddScatter(
	//	"2.Start center position.png",
	//	alg.GetCenters(),
	//	draw.BoxGlyph{},
	//	color.RGBA{65,105,225,255},
	//	6)
	//
	//plot.AddScatter(
	//	"3.End center position.png",
	//	alg.Predict(),
	//	draw.PyramidGlyph{},
	//	color.RGBA{0,100,0,255},
	//	8)

	//plot.CreateScatter(
	//	"4.Data position.png",
	//	gaussian,
	//	draw.CircleGlyph{},
	//	color.RGBA{129,0,0,185},
	//	5)
	//
	//plot.AddScatter(
	//	"5.End center position.png",
	//	model.Centroids,
	//	draw.PyramidGlyph{},
	//	color.RGBA{0,100,0,255},
	//	8)

	clusters := 3
	dimension := 4
	iris_features, _ := readAndPrepareDaten("./iris_features.csv", 4)
	iris_labels, _ := readAndPrepareDate1("./iris_labels.csv")
	plot := new(make_plot.PltStruct)
	//plot.CreateScatter(
	//	"1.Data position.png",
	//	iris_features,
	//	draw.CircleGlyph{},
	//	color.RGBA{129, 0, 0, 185},
	//	5)
	//fmt.Println(len(iris_features[0]))
	alg := new(algorithm_kmeans.KMeans)
	alg.InitAlgorithm(clusters, dimension, 150)

	//centers := make([][]float64, 3)
	//centers[0] = []float64{6, 4, 0, 0}
	//centers[1] = []float64{4, 2, 3, 0}
	//centers[2] = []float64{8, 2, 8, 4}
	//alg.SetCenters(centers)

	//Shuffling the data
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(iris_labels),
		func(i, j int) {
			iris_labels[i], iris_labels[j] = iris_labels[j], iris_labels[i]
			iris_features[i], iris_features[j] = iris_features[j], iris_features[i]
		},
	)
	board := int(len(iris_features) / 4 * 3)
	_, result := alg.Fit(iris_features[:board], iris_labels[:board], true, true)
	//_, result := alg.Fit(iris_features, iris_labels, true, true)

	fmt.Println("KMeans hand made.")
	fmt.Println(
		"Distans: ", result.Metrics,
		"\nTrain accuracy: ", result.Accuracy,
		"\nCounts Points: ", result.CountsPoints,
		"\nCenters: ", result.Centers,
	)

	result_2 := alg.Predict(iris_features[board:])
	fmt.Println(
		"\nPredict labels: ", result_2.Predict,
		"\nTrue labels: ", iris_labels[board:],
		"\nCounts Points: ", result_2.CountsPoints,
		"\nTest accuracy: ", alg.Accuracy(result_2.Predict, iris_labels[board:]),
	)
	//result_2 := alg.Predict(iris_features)
	//fmt.Println(
	//	"\nPredict labels: ", result_2.Predict,
	//	"\nTrue labels: ", iris_labels,
	//	"\nCounts Points: ", result_2.CountsPoints,
	//	"\nTest accuracy: ", alg.Accuracy(result_2.Predict, iris_labels),
	//)

	pl := make([][]float64, len(result.Metrics))
	for i := 0; i < len(result.Metrics); i++ {
		pl[i] = make([]float64, 2)
		pl[i][0] = float64(i + 1)
		pl[i][1] = result.Metrics[i]
	}
	plot.CreateLine(
		"dist plot.png",
		pl,
		color.RGBA{129, 0, 0, 185},
		3)
	pl_1 := make([][]float64, len(result.Accuracy))
	for i := 0; i < len(result.Accuracy); i++ {
		pl_1[i] = make([]float64, 2)
		pl_1[i][0] = float64(i + 1)
		pl_1[i][1] = result.Accuracy[i]
	}
	plot.CreateLine(
		"accuracy plot.png",
		pl_1,
		color.RGBA{129, 0, 0, 185},
		3)


	fmt.Println(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
	fmt.Println("KMeans++ out of the box.")
	model := cluster.NewKMeans(3, 150, iris_features[:board])
	model.Learn()

	fmt.Println("Train Predict: ", model.Guesses())
	fmt.Println("True labels: ", iris_labels[:board])
	fmt.Println("Train accuracy: ", alg.Accuracy(model.Guesses(), iris_labels[:board]))
	fmt.Println("Counts Points: ", consist(model.Guesses()))

	predict := make([]int, len(iris_features[board:]))
	for i := board; i < 150; i++ {
		buf, _ := model.Predict(iris_features[i], false)
		predict[i - board] = int(buf[0])
	}
	fmt.Println("Test Predict: ", predict)
	fmt.Println("True labels: ", iris_labels[board:])
	fmt.Println("Test accuracy: ", alg.Accuracy(predict, iris_labels[board:]))

}

func consist(predict []int) []int {
	count := []int{0, 0, 0}
	for _, v := range predict {
		if v == 0 {
			count[0]++
		}
		if v == 1 {
			count[1]++
		}
		if v == 2 {
			count[2]++
		}
	}
	return count
}

func readAndPrepareDaten(text string, dimension int) ([][]float64, error) {
	byteData, err := ioutil.ReadFile(text)
	if nil != err {
		return nil, err
	}
	rowStrings := strings.Split(string(byteData), "\n")
	floatData := make([][]float64, len(rowStrings))
	for i, v := range rowStrings {
		elts := strings.Split(v, ",")[:dimension]
		floatData[i] = make([]float64, dimension)
		for j, ch := range elts {
			floatData[i][j], _ = strconv.ParseFloat(ch, 64)
		}
	}

	return floatData, nil
}

func readAndPrepareDate1(text string) ([]int, error) {
	byteData, err := ioutil.ReadFile(text)
	if nil != err {
		return nil, err
	}
	stringData := strings.Split(string(byteData), "\n")
	floatData := make([]int, len(stringData))
	for i := range stringData {
		intValue, _ := strconv.ParseInt(stringData[i], 10, 16)
		floatData[i] = int(intValue)
	}
	return floatData, nil
}
