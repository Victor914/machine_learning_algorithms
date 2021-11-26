package algorithm_dbscan

import (
	"fmt"
	dbscan "github.com/Victor914/machine_learning_algorithms/algorithm_dbscan"
	plt "github.com/Victor914/machine_learning_algorithms/make_plot"
	"gonum.org/v1/plot/vg/draw"
	"image/color"
	"io/ioutil"
	"math/rand"
	"strconv"
	"strings"
	"time"
)

func main() {
	//data := [][]float64{{1,1}, {1,1.5},{1.5,1},{1.5,1.5},{2,3},{2,3.5},{2,4},{2.5,3},{2.5,3.5},{2.5,4}}
	alg := new(dbscan.DBSCAN)

	// noisy_circles
	fmt.Println("////////////////////////////////////")
	fmt.Println("noisy_circles")
	alg.InitAlgorithm(3, 0.1)
	feat, labels, _ := readAndPrepareFeatLabels("./data/noisy_circles.txt", 2)
	for i := range labels {
		labels[i]++
		if labels[i] == 1 {labels[i] = 2} else {labels[i] = 1}
	}
	out := alg.Fit(feat)
	fmt.Println("Accuracy:\t", alg.Accuracy(out.Predict, labels))
	fmt.Println("Predict:\t", out.Predict)
	fmt.Println("True:\t\t", labels)
	buildPlot(feat, out.Predict, out.CountClusters, "Noisy circles dataset")

	// noisy_moons
	fmt.Println("////////////////////////////////////")
	fmt.Println("noisy_moons")
	alg.InitAlgorithm(3, 0.1)
	feat, labels, _ = readAndPrepareFeatLabels("./data/noisy_moons.txt", 2)
	for i := range labels {
		labels[i]++
		if labels[i] == 1 {labels[i] = 2} else {labels[i] = 1}
	}
	out = alg.Fit(feat)
	fmt.Println("Accuracy:\t", alg.Accuracy(out.Predict, labels))
	fmt.Println("Predict:\t", out.Predict)
	fmt.Println("True:\t\t", labels)
	buildPlot(feat, out.Predict, out.CountClusters, "Noisy moons dataset")

	// blobs
	fmt.Println("////////////////////////////////////")
	fmt.Println("blobs")
	alg.InitAlgorithm(3, 1)
	feat, labels, _ = readAndPrepareFeatLabels("./data/blobs.txt", 2)

	for i := range labels {
		labels[i]++
	}
	out = alg.Fit(feat)
	fmt.Println("Accuracy:\t", alg.Accuracy(out.Predict, labels))
	fmt.Println("Predict:\t", out.Predict)
	fmt.Println("True:\t\t", labels)
	buildPlot(feat, out.Predict, out.CountClusters, "Blobs dataset")

}

// readAndPrepareFeatLabels input features and labels from file
func readAndPrepareFeatLabels(text string, dimension int) ([][]float64, []int64, error) {
	byteData, err := ioutil.ReadFile(text)
	if nil != err {
		return nil, nil, err
	}
	rowStrings := strings.Split(string(byteData), "\n")
	featFloatData := make([][]float64, len(rowStrings))
	labelsFloatData := make([]int64, len(rowStrings))
	for i, v := range rowStrings {
		elts := strings.Split(v, "\t")[:dimension + 1]
		featFloatData[i] = make([]float64, dimension)
		for j, ch := range elts {
			if j == dimension {
				intValue, _ := strconv.ParseInt(ch, 10, 16)
				labelsFloatData[i] = intValue
			} else {
				featFloatData[i][j], _ = strconv.ParseFloat(ch, 64)
			}
		}
	}
	return featFloatData, labelsFloatData, nil
}


func buildPlot(feat [][]float64, labels []int64, countClusters int64, nameFile string) {
	plot := new(plt.PltStruct)
	buff := make(map[int64][][]float64)
	for indPoint, point := range feat {
		buff[labels[indPoint]] = append(buff[labels[indPoint]], point)
	}
	rand.Seed(time.Now().UnixNano())
	plot.CreateScatter(
		nameFile+".png",
		buff[1],
		draw.CircleGlyph{},
		color.RGBA{
			uint8(int8(rand.Intn(256))),
			uint8(int8(rand.Intn(256))),
			uint8(int8(rand.Intn(256))),
			230},
		5)
	for i := 2; i <= int(countClusters); i++{
		plot.AddScatter(
			nameFile+".png",
			buff[int64(i)],
			draw.CircleGlyph{},
			color.RGBA{
				uint8(int8(rand.Intn(256))),
				uint8(int8(rand.Intn(256))),
				uint8(int8(rand.Intn(256))),
				230},
			5)
	}

}
