package algorithm_cart

import (
	"fmt"
	"math"
	"math/rand"
)

type CART struct {
	depth     int
	minCount  int
	countFold int
	Scores    []float64
	Trees     []*Tree
}

func (k *CART) InitAlgorithm(depth int, minCunt int, countFold int) {
	k.depth = depth
	k.minCount = minCunt
	k.countFold = countFold
}

//Accuracy calculates the quality metric
func (k *CART) Accuracy(predict []int64, labels []int64) float64 {
	metrica := 0
	for i := range predict {
		if labels[i] == predict[i] {
			metrica++
		}
	}
	return float64(metrica) / float64(len(labels))
}

func (k *CART) CrossValSplit(data [][]float64) [][][]float64 {
	sizeFold := len(data) / k.countFold
	folds := make([][][]float64, 0)
	copyData := make([][]float64, len(data))
	for i := range data {
		copyData[i] = make([]float64, len(data[i]))
		copy(copyData[i], data[i])
	}
	for i := 0; i < k.countFold; i++ {
		fold := make([][]float64, sizeFold)
		for ind := range fold {
			randInd := rand.Intn(len(copyData))
			fold[ind] = make([]float64, len(copyData[randInd]))
			copy(fold[ind], copyData[randInd])
			copyData = append(copyData[:randInd], copyData[randInd+1:]...)
		}
		folds = append(folds, fold)
	}
	return folds
}

type Tree struct {
	mainNode *node
}

type node struct {
	sheet bool
	left  *node
	right *node
	value float64
	index int64
	label int64
}

type data struct {
	x [][]float64
	y []int64
}

func (k *CART) CrossValidationScore(data [][]float64) float64 {
	folds := k.CrossValSplit(data)
	for indFold, fold := range folds {
		trainSet := make([][]float64, 0)
		for ind := 0; ind < indFold; ind++ {
			trainSet = append(trainSet, folds[ind]...)
		}
		for ind := indFold + 1; ind < len(folds); ind++ {
			trainSet = append(trainSet, folds[ind]...)
		}
		testSet := fold

		train := k.splitOnFeaturesAndLabels(trainSet)
		test := k.splitOnFeaturesAndLabels(testSet)

		tree := k.makeTree(train)
		predict := tree.Predict(test.x)
		k.Trees = append(k.Trees, tree)
		k.Scores = append(k.Scores, k.Accuracy(predict, test.y))
	}
	sumScores := 0.0
	for _, el := range k.Scores {
		sumScores += el
	}
	return sumScores / float64(len(k.Scores))
}

func (k *CART) splitOnFeaturesAndLabels(allData [][]float64) *data {
	labels := make([]int64, 0)
	features := make([][]float64, 0)
	for _, row := range allData {
		labels = append(labels, int64(row[len(row)-1]))
		features = append(features, row[:len(row)-1])
	}
	return &data{
		x: features,
		y: labels,
	}
}

func (k *CART) makeTree(train *data) *Tree {
	classes := make(map[int64]bool)
	for _, el := range train.y {
		classes[el] = true
	}
	tree := new(Tree)
	tree.mainNode = k.makeNode(train, classes, 1)
	return tree
}

func (k *CART) makeNode(train *data, classes map[int64]bool, depth int) *node {
	if depth >= k.depth {
		return k.makeSheet(train)
	}
	curNode, left, right := k.choiceSplit(classes, train)

	if len(left.y) <= k.minCount {
		curNode.left = k.makeSheet(left)
	} else {
		curNode.left = k.makeNode(left, classes, depth+1)
	}
	if len(right.y) <= k.minCount {
		curNode.right = k.makeSheet(right)
	} else {
		curNode.right = k.makeNode(right, classes, depth+1)
	}
	return curNode
}

func (k *CART) makeSheet(train *data) *node {
	classes := make(map[int64]int64)
	for _, el := range train.y {
		classes[el] += 1
	}
	var label, value int64 = 0, 0
	iter := 0
	for l, v := range classes {
		if iter == 0 || v > value {
			value = v
			label = l
		}
		iter += 1
	}
	return &node{
		sheet: true,
		left:  nil,
		right: nil,
		label: label,
	}
}

func (k *CART) choiceSplit(classes map[int64]bool, train *data) (*node, *data, *data) {
	var left, right *data
	value, score := 0.0, 0.0
	var index int64 = 0

	for indFeat := range train.x[0] {
		for indRow, row := range train.x {
			left, right = k.testSplit(train, indFeat, row[indFeat])
			gini := k.indexGini(left.y, right.y, classes)
			if indRow == 0 && indFeat == 0 || gini < score {
				score, value, index = gini, row[indFeat], int64(indFeat)
			}
		}
	}
	return &node{
		sheet: false,
		value: value,
		index: index,
	}, left, right
}

func (k *CART) indexGini(yLeft []int64, yRight []int64, classes map[int64]bool) float64 {
	percentClass := func(class int64, labels []int64) float64 {
		if len(labels) != 0 {
			count := 0
			for _, el := range labels {
				if el == class {
					count++
				}
			}
			return float64(count) / float64(len(labels))
		} else {
			return 0.0
		}
	}

	gini := 0.0
	for cl := range classes {
		prL := percentClass(cl, yLeft)
		prR := percentClass(cl, yRight)
		gini += prR * (1 - prR)
		gini += prL * (1 - prL)
	}
	return gini
}

func (k *CART) testSplit(allData *data, index int, value float64) (*data, *data) {
	xLeft, xRight := make([][]float64, 0), make([][]float64, 0)
	yLeft, yRight := make([]int64, 0), make([]int64, 0)
	for ind := range allData.x {
		if allData.x[ind][index] < value {
			xLeft = append(xLeft, allData.x[ind])
			yLeft = append(yLeft, allData.y[ind])
		} else {
			xRight = append(xRight, allData.x[ind])
			yRight = append(yRight, allData.y[ind])
		}
	}
	return &data{x: xLeft, y: yLeft}, &data{x: xRight, y: yRight}
}

func (t *Tree) Predict(features [][]float64) []int64 {
	predict := make([]int64, 0)
	for _, feat := range features {
		predict = append(predict, t.mainNode.predict(feat))
	}
	return predict
}

func (t *node) predict(feat []float64) int64 {
	if t.value < feat[t.index] {
		if t.sheet {
			return t.label
		} else {
			return t.left.predict(feat)
		}
	} else {
		if t.sheet {
			return t.label
		} else {
			return t.right.predict(feat)
		}
	}
}

func (t *Tree) PrintTree() {
	queue := make([]*node, 0)
	queue = append(queue, t.mainNode)
	i := 1.0
	l := 0.0
	fmt.Println("Decision Tree")
	fmt.Println("Level 0")
	for len(queue) != 0 {
	}
	if queue[0].sheet {
		fmt.Print("label: ", queue[0].label, "    ")

	} else {
		fmt.Print("index: ", queue[0].index, " value: ", queue[0].value, "    ")
		queue = append(queue, queue[0])
	}
	if math.Log2(i) == l {
		fmt.Println()
		fmt.Println("Level ", int(l + 1))
		l += 1
	}
	i += 1
	queue = queue[1:]
}
