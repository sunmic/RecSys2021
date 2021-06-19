package main

// #cgo LDFLAGS: -lm -ligraph
// #include <igraph/igraph.h>
// #include <stdio.h>
// #include "igraph_ext.h"
import "C"
import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Config struct {
	batchSize           int
	nWorkers            int
	useWeigths          bool
	startContentSamples int
	contentSamples      int
	socialSamples       []int
	tweetEdgelistPath   string
	followEdgelistPath  string
	destDir             string
}

func parseGraph(path string, weigths bool, directed bool) *C.igraph_t {
	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	var graph *C.igraph_t = C.igraph_alloc()
	fstruct := C.fdopen(C.int(file.Fd()), C.CString("r"))

	var errCode C.int
	if weigths {
		C.igraph_set_attribute_table(&C.igraph_cattribute_table)
		errCode = C.igraph_read_graph_weighted_edgelist(graph, fstruct, 0, booltoint(directed), WeightAttributeName)
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

func parseArgs() Config {
	flag.Usage = func() {
		fmt.Printf("Usage: %v follow-edgelist tweet-edgelist dest-dir\n", os.Args[0])
		flag.PrintDefaults()
	}
	batchSize := flag.Int("batch", 1000, "Number of neighbourhoods in single batch file")
	nWorkers := flag.Int("nproc", 1, "Number of workers")
	startContentSamples := flag.Int("start-content", 30, "Number of content samples per start node")
	contentSamples := flag.Int("content", 10, "Number of content samples per non-start node")
	levelSamplesRaw := flag.String("social", "10,5", "Number of social samples at each level (comma separated)")
	useWeights := flag.Bool("weights", false, "Adjacency lists has third parameter with edge weight. Weight will be taken into account")

	flag.Parse()
	if flag.NArg() < 3 {
		flag.Usage()
		os.Exit(1)
	}

	followEdgelistPath := flag.Arg(0)
	tweetEdgelistPath := flag.Arg(1)
	destDir := flag.Arg(2)
	levelSamples := parseLevelSamples(*levelSamplesRaw)

	config := Config{
		batchSize:           *batchSize,
		nWorkers:            *nWorkers,
		useWeigths:          *useWeights,
		startContentSamples: *startContentSamples,
		contentSamples:      *contentSamples,
		socialSamples:       levelSamples,
		tweetEdgelistPath:   tweetEdgelistPath,
		followEdgelistPath:  followEdgelistPath,
		destDir:             destDir,
	}

	return config
}

func main() {
	config := parseArgs()

	log.Printf("igraph thread safety flag: %v", C.IGRAPH_THREAD_SAFE)
	if config.nWorkers > 1 && C.IGRAPH_THREAD_SAFE == 0 {
		log.Fatalln("igraph not build with thread safety. Recompile library or use nproc = 1. Failing.")
	}

	log.Println("Parsing tweets graph...")
	tweetsGraph := parseGraph(config.tweetEdgelistPath, config.useWeigths, true)
	log.Println("Parsing followers graph...")
	followersGraph := parseGraph(config.followEdgelistPath, false, false)

	rand.Seed(time.Now().UnixNano())

	var wg sync.WaitGroup
	channel := make(chan int32)
	task := Task{wg: &wg, batchSize: config.batchSize, destDir: config.destDir, useEdgeTypes: config.useWeigths}
	go task.SubmitNodes(followersGraph, channel)
	log.Printf("Starting %v workers", config.nWorkers)
	for i := 0; i < config.nWorkers; i++ {
		go task.GenerateNeighbourhoods(followersGraph, tweetsGraph, config.socialSamples, config.contentSamples, config.startContentSamples, channel, i)
		wg.Add(1)
	}

	wg.Wait()

	log.Println("Done!")
	C.igraph_destroy(followersGraph)
	C.igraph_destroy(tweetsGraph)
}
