package main

// #cgo pkg-config: igraph
// #cgo LDFLAGS: -lm
// #include <igraph.h>
// #include <stdio.h>
// #include "igraph_ext.h"
import "C"
import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	models "igraph/proto"

	pb "github.com/cheggaaa/pb/v3"
	"google.golang.org/protobuf/proto"
)

const WeightAttributeName = "weight"

func booltoint(in bool) C.int {
	if in {
		return C.int(1)
	}
	return C.int(0)
}

func randomKSample(max int, population int, result *[]int) {
	if population <= 0 || max < population {
		panic("Invalid population")
	}

	var pickMap = make(map[int]bool)
	var randomValue int
	for len(*result) < population {
		randomValue = rand.Intn(max)
		if !pickMap[randomValue] {
			pickMap[randomValue] = true
			*result = append(*result, randomValue)
		}
	}
}

type NNSampling struct {
	visited     map[int32]bool
	sources     []int32
	targets     []int32
	edgeTypes   []int32
	startVertex int32
	// Temporary storage
	neighbourhoodBuffer []int32
	randomMask          []int
}

func (sampling *NNSampling) Print() {
	fmt.Printf("Visited: %v\n", sampling.visited)
	fmt.Printf("Sources: %v\n", sampling.sources)
	fmt.Printf("Targets: %v\n", sampling.targets)
	if len(sampling.edgeTypes) != 0 {
		fmt.Printf("Edge types: %v\n", sampling.edgeTypes)
	}
	fmt.Printf("Buffer:  %v\n", sampling.neighbourhoodBuffer)
	fmt.Printf("Mask:    %v\n", sampling.randomMask)
}

func (sampling *NNSampling) Reset() {
	sampling.visited = make(map[int32]bool)
	sampling.sources = sampling.sources[:0]
	sampling.targets = sampling.targets[:0]
	sampling.edgeTypes = sampling.edgeTypes[:0]
	sampling.neighbourhoodBuffer = sampling.neighbourhoodBuffer[:0]
	sampling.randomMask = sampling.randomMask[:0]
}

func (state *NNSampling) SampleImmediateNeighbourhood(graph *C.igraph_t, vertex int32, samples int, duplicates bool, clearBuffers bool, allEdges bool, weights bool) {
	var vs C.igraph_vs_t
	var vit C.igraph_vit_t
	var eid C.igraph_integer_t
	var eval C.igraph_real_t

	// set self as visited
	state.visited[vertex] = true

	// if allEdges IGRAPH_ALL else IGRAPH_OUT
	var mode C.igraph_neimode_t
	if allEdges {
		mode = C.IGRAPH_ALL
	} else {
		mode = C.IGRAPH_OUT
	}

	// check if weight attribute is present
	var hasWeightAttribute = weights

	// find non-visited neighbours using igraph
	C.igraph_vs_adj(&vs, C.int(vertex), mode)
	C.igraph_vit_create(graph, vs, &vit)
	for C.igraph_vit_end(&vit) == 0 {
		adjVertex := int32(C.igraph_vit_get(&vit))

		if duplicates || !state.visited[adjVertex] {
			state.neighbourhoodBuffer = append(state.neighbourhoodBuffer, adjVertex)
		}

		C.igraph_vit_next(&vit)
	}

	if samples >= len(state.neighbourhoodBuffer) {
		// no need to sample
		for index := 0; index < len(state.neighbourhoodBuffer); index++ {
			state.randomMask = append(state.randomMask, index)
		}
	} else {
		// create random index sequence and get those neighbours
		randomKSample(len(state.neighbourhoodBuffer), samples, &state.randomMask)
	}

	// update state with new edges
	for _, index := range state.randomMask {
		adjVertex := state.neighbourhoodBuffer[index]
		state.visited[adjVertex] = true
		state.sources = append(state.sources, vertex)
		state.targets = append(state.targets, adjVertex)
		if hasWeightAttribute {
			C.igraph_get_eid(graph, &eid, C.int(vertex), C.int(adjVertex), booltoint(true), booltoint(false))
			if eid != -1 {
				eval = C.igraph_edge_weight_attribute(graph, eid)
				state.edgeTypes = append(state.edgeTypes, int32(eval))
			}
		}
	}

	if clearBuffers {
		// clear keeping capacity
		state.neighbourhoodBuffer = state.neighbourhoodBuffer[:0]
		state.randomMask = state.randomMask[:0]
	}

	// clean
	C.igraph_vs_destroy(&vs)
	C.igraph_vit_destroy(&vit)
}

func (state *NNSampling) SampleNeighbourhood(graph *C.igraph_t, vertex int32, levelSamples []int, level int) {
	if level >= len(levelSamples) {
		return
	}

	// clear keeping capacity
	state.neighbourhoodBuffer = state.neighbourhoodBuffer[:0]
	state.randomMask = state.randomMask[:0]

	// sample immediate neighbours
	samples := levelSamples[level]
	state.SampleImmediateNeighbourhood(graph, vertex, samples, false, false, true, false)

	// sample neighbourhoods of neighbours if needed
	if level+1 < len(levelSamples) {
		sampledNeighbourhood := make([]int32, len(state.randomMask))
		for i, index := range state.randomMask {
			adjVertex := state.neighbourhoodBuffer[index]
			sampledNeighbourhood[i] = adjVertex
		}

		for _, adjVertex := range sampledNeighbourhood {
			state.SampleNeighbourhood(graph, adjVertex, levelSamples, level+1)
		}
	}
}

func pushState(socialState *NNSampling, contentState *NNSampling, batch *models.SocialNetworkBatch) {
	// create user neighbourhood
	userNodes := make([]int32, 0, len(socialState.visited))
	for vertex, present := range socialState.visited {
		if !present {
			continue
		}
		userNodes = append(userNodes, vertex)
	}
	sourcesCopy := make([]int32, len(socialState.sources))
	copy(sourcesCopy, socialState.sources)
	targetsCopy := make([]int32, len(socialState.targets))
	copy(targetsCopy, socialState.targets)

	followersNeighbourhood := models.Neighbourhood{
		Start:           proto.Int32(socialState.startVertex),
		Nodes:           userNodes,
		EdgeIndexSource: sourcesCopy,
		EdgeIndexTarget: targetsCopy,
	}

	// create tweet neighbourhood
	tweetNodes := make([]int32, 0, len(contentState.visited))
	for vertex, present := range contentState.visited {
		if !present {
			continue
		}
		tweetNodes = append(tweetNodes, vertex)
	}
	sourcesCopy = make([]int32, len(contentState.sources))
	copy(sourcesCopy, contentState.sources)
	targetsCopy = make([]int32, len(contentState.targets))
	copy(targetsCopy, contentState.targets)
	var edgeTypes *models.Neighbourhood_EdgeTypes = nil
	if len(contentState.edgeTypes) != 0 {
		edgeTypes = &models.Neighbourhood_EdgeTypes{}
		for _, edgeAttr := range contentState.edgeTypes {
			attributes := &models.Neighbourhood_EdgeType{}
			attributes.Seen = proto.Bool((edgeAttr & 0x1) != 0)
			attributes.Like = proto.Bool((edgeAttr & 0x2) != 0)
			attributes.Reply = proto.Bool((edgeAttr & 0x4) != 0)
			attributes.Retweet = proto.Bool((edgeAttr & 0x8) != 0)
			attributes.RetweetComment = proto.Bool((edgeAttr & 0x10) != 0)
			edgeTypes.Attributes = append(edgeTypes.Attributes, attributes)
		}
	}

	tweetNeighbourhood := models.Neighbourhood{
		Start:           proto.Int32(contentState.startVertex),
		Nodes:           tweetNodes,
		EdgeIndexSource: sourcesCopy,
		EdgeIndexTarget: targetsCopy,
		EdgeTypes:       edgeTypes,
	}

	socialNeighbourhood := models.SocialNetworkNeighbourhood{
		SocialNeighbourhood:  &followersNeighbourhood,
		ContentNeighbourhood: &tweetNeighbourhood,
	}
	batch.Elements = append(batch.Elements, &socialNeighbourhood)
}

type Task struct {
	wg           *sync.WaitGroup
	pbar         *pb.ProgressBar
	batchSize    int
	useEdgeTypes bool
	destDir      string
}

func (task *Task) SaveBatch(batch *models.SocialNetworkBatch, iteration int, work int) bool {
	out, err := proto.Marshal(batch)
	if err != nil {
		log.Printf("Error while serializing batch: %v\n", err)
		return false
	}

	filename := fmt.Sprintf("batch_%v_%v", work, iteration)
	filePath := path.Join(task.destDir, filename)
	if err := ioutil.WriteFile(filePath, out, 0644); err != nil {
		log.Printf("Error while saving batch: %v\n", err)
		return false
	}

	return true
}

// consumer procedure, just consume new vertices and calculate neighbourhoods
func (task *Task) GenerateNeighbourhoods(followerGraph *C.igraph_t, tweetsGraph *C.igraph_t, levelSamples []int, contentSamples int, channel <-chan int32, workIndex int) {
	defer task.wg.Done()

	var socialState = NNSampling{visited: make(map[int32]bool)}
	var contentState = NNSampling{visited: make(map[int32]bool)}
	var batch = &models.SocialNetworkBatch{}
	var iteration int

	for vertex := range channel {
		socialState.startVertex = vertex
		contentState.startVertex = vertex
		socialState.SampleNeighbourhood(followerGraph, vertex, levelSamples, 0)
		for socialVertex, present := range socialState.visited {
			if !present {
				continue
			}
			contentState.SampleImmediateNeighbourhood(tweetsGraph, socialVertex, contentSamples, true, true, false, task.useEdgeTypes)
			contentState.visited[socialVertex] = false
		}

		pushState(&socialState, &contentState, batch)
		socialState.Reset()
		contentState.Reset()

		if len(batch.Elements) == task.batchSize {
			task.SaveBatch(batch, iteration+1, workIndex)
			proto.Reset(batch)
		}

		task.pbar.Increment()
		iteration += 1
	}

	if len(batch.Elements) != 0 {
		task.SaveBatch(batch, iteration, workIndex)
		proto.Reset(batch)
	}
}

// producer procedure, produce nodes to calculate neighbourhoods for
func (task *Task) SubmitNodes(followerGraph *C.igraph_t, channel chan<- int32) {
	var vs C.igraph_vs_t
	var vit C.igraph_vit_t

	vertexCount := int64(C.igraph_vcount(followerGraph))
	pbar := pb.New64(vertexCount)
	pbar.SetRefreshRate(time.Millisecond * 300)
	pbar.Start()
	task.pbar = pbar

	C.igraph_vs_all(&vs)
	C.igraph_vit_create(followerGraph, vs, &vit)
	for C.igraph_vit_end(&vit) == 0 {
		vertex := int32(C.igraph_vit_get(&vit))
		channel <- vertex
		C.igraph_vit_next(&vit)
	}

	C.igraph_vs_destroy(&vs)
	C.igraph_vit_destroy(&vit)
	close(channel)
}

func parseGraph(path string, nCol bool, directed bool) *C.igraph_t {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	var graph *C.igraph_t = C.igraph_alloc()
	fstruct := C.fdopen(C.int(file.Fd()), C.CString("r"))

	var errCode C.int
	if nCol {
		C.igraph_set_attribute_table(&C.igraph_cattribute_table)
		errCode = C.igraph_read_graph_ncol(graph, fstruct, nil, booltoint(false), C.IGRAPH_ADD_WEIGHTS_YES, booltoint(directed))
	} else {
		errCode = C.igraph_read_graph_edgelist(graph, fstruct, 0, booltoint(directed))
	}

	if errCode == C.IGRAPH_PARSEERROR {
		log.Fatal(err)
	}

	vCount := int(C.igraph_vcount(graph))
	eCount := int(C.igraph_ecount(graph))
	log.Printf("Created graph with %v vertices and %v edges", vCount, eCount)

	return graph
}

func parseLevelSamples(s string) []int {
	strLevels := strings.Split(s, ",")
	levels := make([]int, len(strLevels))
	for i, str := range strLevels {
		level, err := strconv.Atoi(str)
		if err != nil {
			log.Fatal(err)
		}

		levels[i] = level
	}

	return levels
}

func parseArgs() (int, int, bool, int, []int, string, string, string) {
	flag.Usage = func() {
		fmt.Printf("Usage: %v follow-edgelist tweet-edgelist dest-dir\n", os.Args[0])
		flag.PrintDefaults()
	}
	batchSize := flag.Int("b", 1000, "Number of neighbourhoods in single batch file")
	nWorkers := flag.Int("nproc", 1, "Number of workers")
	contentSamples := flag.Int("c", 20, "Number of content samples per node")
	levelSamplesRaw := flag.String("s", "10,5", "Number of social samples at each level (comma separated)")
	ncolFormat := flag.Bool("ncol", false, "Adjacency lists are in ncol format. Weight will be taken into account")

	flag.Parse()
	if flag.NArg() < 3 {
		flag.Usage()
		os.Exit(1)
	}

	followEdgelistPath := flag.Arg(0)
	tweetEdgelistPath := flag.Arg(1)
	destDir := flag.Arg(2)
	levelSamples := parseLevelSamples(*levelSamplesRaw)

	return *batchSize, *nWorkers, *ncolFormat, *contentSamples, levelSamples, tweetEdgelistPath, followEdgelistPath, destDir
}

func main() {
	batchSize, nWorkers, ncolFormat, contentSamples, levelSamples, tweetEdgelistPath, followEdgelistPath, destDir := parseArgs()

	log.Printf("igraph thread safety flag: %v", C.IGRAPH_THREAD_SAFE)
	if nWorkers > 1 && C.IGRAPH_THREAD_SAFE == 0 {
		log.Fatalln("igraph not build with thread safety. Recompile library or use nproc = 1. Failing.")
	}

	log.Println("Parsing tweets graph...")
	tweetsGraph := parseGraph(tweetEdgelistPath, ncolFormat, true)
	log.Println("Parsing followers graph...")
	followersGraph := parseGraph(followEdgelistPath, false, false)

	rand.Seed(time.Now().UnixNano())

	var wg sync.WaitGroup
	channel := make(chan int32)
	task := Task{wg: &wg, batchSize: batchSize, destDir: destDir, useEdgeTypes: ncolFormat}
	go task.SubmitNodes(followersGraph, channel)
	log.Printf("Starting %v workers", nWorkers)
	for i := 0; i < nWorkers; i++ {
		go task.GenerateNeighbourhoods(followersGraph, tweetsGraph, levelSamples, contentSamples, channel, i)
		wg.Add(1)
	}

	wg.Wait()

	log.Println("Done!")
	C.igraph_destroy(followersGraph)
	C.igraph_destroy(tweetsGraph)
}
