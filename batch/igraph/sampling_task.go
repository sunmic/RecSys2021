package main

// #cgo LDFLAGS: -lm -ligraph
// #include <igraph/igraph.h>
// #include "igraph_ext.h"
import "C"
import (
	"fmt"
	"io/ioutil"
	"log"
	"path"
	"sync"
	"time"

	models "igraph/proto"

	pb "github.com/cheggaaa/pb/v3"
	"google.golang.org/protobuf/proto"
)

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
func (task *Task) GenerateNeighbourhoods(followerGraph *C.igraph_t, tweetsGraph *C.igraph_t, socialSamples []int, contentSamples int, startContentSamples int, channel <-chan int32, workIndex int) {
	defer task.wg.Done()

	var socialState = NNSampling{visited: make(map[int32]bool)}
	var contentState = NNSampling{visited: make(map[int32]bool)}
	var batch = &models.SocialNetworkBatch{}
	var iteration int

	for vertex := range channel {
		socialState.startVertex = vertex
		contentState.startVertex = vertex
		socialState.SampleNeighbourhood(followerGraph, vertex, socialSamples, 0, false)
		for socialVertex, present := range socialState.visited {
			if !present {
				continue
			}

			var samples = contentSamples
			if socialVertex == vertex {
				samples = startContentSamples
			}
			contentState.SampleImmediateNeighbourhood(tweetsGraph, socialVertex, samples, true, true, false, task.useEdgeTypes, false, task.useEdgeTypes)
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
