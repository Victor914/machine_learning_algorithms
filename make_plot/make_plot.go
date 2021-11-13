	package make_plot

import (
	"bufio"
	"fmt"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"image/color"
	"log"
	"os"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
)

type PltStruct struct {
	plt *plot.Plot
}

type xy struct{ x, y float64 }

func readData(path string) (plotter.XYs, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var xys plotter.XYs
	s := bufio.NewScanner(f)
	for s.Scan() {
		var x, y float64
		_, err := fmt.Sscanf(s.Text(), "%f,%f", &x, &y)
		if err != nil {
			log.Printf("discarding bad data point %q: %v", s.Text(), err)
			continue
		}
		xys = append(xys, struct{ X, Y float64 }{x, y})
	}
	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("could not scan: %v", err)
	}
	return xys, nil
}

func floatToXYs(mas [][]float64) plotter.XYs {
	var xys plotter.XYs
	for _, v := range mas {
		xys = append(xys, struct{ X, Y float64 }{v[0], v[1]})
	}
	return xys
}

// CreateScatter creates scatter plot.
func (p *PltStruct) CreateScatter(
	fileOut string,
	data [][]float64,
	marker draw.GlyphDrawer,
	rgba color.RGBA,
	radius vg.Length,
) error {
	f, err := os.Create(fileOut)
	if err != nil {
		return fmt.Errorf("could not create %s: %v", fileOut, err)
	}
	p.plt = plot.New()
	s, err := plotter.NewScatter(floatToXYs(data))
	if err != nil {
		return fmt.Errorf("could not create scatter: %v", err)
	}
	s.GlyphStyle.Shape = marker
	s.Radius = radius
	s.Color = rgba
	p.plt.Add(s)
	wt, err := p.plt.WriterTo(512, 512, "png")
	if err != nil {
		return fmt.Errorf("could not create writer: %v", err)
	}
	if _, err = wt.WriteTo(f); err != nil {
		return fmt.Errorf("could not write to %s: %v", fileOut, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("could not close %s: %v", fileOut, err)
	}
	return nil
}

// CreateLine creates liners graph.
func (p *PltStruct) CreateLine(
	fileOut string,
	data [][]float64,
	rgba color.RGBA,
	width vg.Length,
) error {
	f, err := os.Create(fileOut)
	if err != nil {
		return fmt.Errorf("could not create %s: %v", fileOut, err)
	}
	p.plt = plot.New()
	l, err := plotter.NewLine(floatToXYs(data))
	if err != nil {
		return fmt.Errorf("could not create line: %v", err)
	}
	l.Width = width
	l.Color = rgba
	p.plt.Add(l)
	wt, err := p.plt.WriterTo(512, 512, "png")
	if err != nil {
		return fmt.Errorf("could not create writer: %v", err)
	}
	if _, err = wt.WriteTo(f); err != nil {
		return fmt.Errorf("could not write to %s: %v", fileOut, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("could not close %s: %v", fileOut, err)
	}
	return nil
}

// AddLine adds liners graph in canvas
func (p *PltStruct) AddLine(
	fileOut string,
	data [][]float64,
	rgba color.RGBA,
	width vg.Length,
) error {
	f, err := os.Create(fileOut)
	if err != nil {
		return fmt.Errorf("could not create %s: %v", fileOut, err)
	}
	l, err := plotter.NewLine(floatToXYs(data))
	if err != nil {
		return fmt.Errorf("could not create line: %v", err)
	}
	l.Width = width
	l.Color = rgba
	p.plt.Add(l)
	wt, err := p.plt.WriterTo(512, 512, "png")
	if err != nil {
		return fmt.Errorf("could not create writer: %v", err)
	}
	if _, err = wt.WriteTo(f); err != nil {
		return fmt.Errorf("could not write to %s: %v", fileOut, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("could not close %s: %v", fileOut, err)
	}
	return nil
}

// AddScatter adds scatter plot in canvas
func (p *PltStruct) AddScatter(
	fileOut string,
	data [][]float64,
	marker draw.GlyphDrawer,
	rgba color.RGBA,
	radius vg.Length,
) error {

	f, err := os.Create(fileOut)
	if err != nil {
		return fmt.Errorf("could not create %s: %v", fileOut, err)
	}
	s, err := plotter.NewScatter(floatToXYs(data))
	if err != nil {
		return fmt.Errorf("could not create scatter: %v", err)
	}
	s.GlyphStyle.Shape = marker
	s.Radius = radius
	s.Color = rgba
	p.plt.Add(s)
	wt, err := p.plt.WriterTo(512, 512, "png")
	if err != nil {
		return fmt.Errorf("could not create writer: %v", err)
	}
	if _, err = wt.WriteTo(f); err != nil {
		return fmt.Errorf("could not write to %s: %v", fileOut, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("could not close %s: %v", fileOut, err)
	}
	return nil
}
