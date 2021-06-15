package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/cheggaaa/pb/v3"
)

func parseHeader(header []string, sourceTag string, targetTag string) (int, int, error) {
	var sourceIndex, targetIndex int = -1, -1
	for i, tag := range header {
		if tag == sourceTag {
			sourceIndex = i
		}
		if tag == targetTag {
			targetIndex = i
		}

		if sourceIndex >= 0 && targetIndex >= 0 {
			break
		}
	}

	var err error = nil
	if sourceIndex == -1 {
		err = errors.New("Source tag not found: " + sourceTag)
	}
	if targetIndex == -1 {
		err = errors.New("Target tag not found: " + targetTag)
	}

	return sourceIndex, targetIndex, err
}

func csvReader(reader io.Reader, delimiter rune) *csv.Reader {
	csvReader := csv.NewReader(reader)
	csvReader.Comma = delimiter
	csvReader.TrimLeadingSpace = true
	csvReader.ReuseRecord = true
	csvReader.FieldsPerRecord = 0

	return csvReader
}

func fileSize(file *os.File) int64 {
	info, err := file.Stat()
	if err != nil {
		log.Fatal(err)
	}
	return info.Size()
}

func fillEdges(edgeMap *map[string]int, filePath string, fileIndex int, sourceTag string, targetTag string, delimiter rune) {
	fileReader, err := os.Open(filePath)
	if err != nil {
		log.Fatal(err)
	}

	reader := csvReader(fileReader, delimiter)

	headerRecord, err := reader.Read()
	if err != nil {
		log.Fatalf("Could not parse header: %v", err)
	}

	sourceIndex, targetIndex, err := parseHeader(headerRecord, sourceTag, targetTag)
	if err != nil {
		log.Fatal(err)
	}

	pbar := pb.New64(fileSize(fileReader))
	pbar.Start()
	var stringBuffer bytes.Buffer
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Println(err)
			continue
		}
		stringBuffer.WriteString(record[sourceIndex])
		stringBuffer.WriteString(" ")
		stringBuffer.WriteString(record[targetIndex])
		key := stringBuffer.String()
		stringBuffer.Reset()
		if bitset, ok := (*edgeMap)[key]; ok {
			bitset = bitset | (1 << fileIndex)
			(*edgeMap)[key] = bitset
		} else {
			(*edgeMap)[key] = (1 << fileIndex)
		}

		pos, _ := fileReader.Seek(0, os.SEEK_CUR)
		pbar.SetCurrent(pos)
	}

	pbar.Finish()
	fileReader.Close()
}

func writeEdgeList(edgeMap map[string]int, outputFilePath string, flushing int64) {
	fileWriter, err := os.OpenFile(outputFilePath, os.O_CREATE|os.O_WRONLY, 0744)
	if err != nil {
		log.Fatal(err)
	}

	var it int64 = 0
	writer := bufio.NewWriter(fileWriter)
	pbar := pb.New(len(edgeMap))
	pbar.Start()
	var stringBuffer bytes.Buffer
	for key, value := range edgeMap {
		stringBuffer.WriteString(key)
		stringBuffer.WriteRune(' ')
		stringBuffer.WriteString(strconv.FormatInt(int64(value), 10))
		stringBuffer.WriteRune('\n')
		line := stringBuffer.String()
		stringBuffer.Reset()
		_, err = writer.WriteString(line)
		if err != nil {
			log.Fatal(err)
		}

		if it%flushing == 0 {
			writer.Flush()
		}

		it += 1
		pbar.SetCurrent(it)
	}
	writer.Flush()

	pbar.Finish()
	fileWriter.Close()
}

func main() {
	delimiterString := flag.String("delimiter", ",", "Input CSV delimiter")
	sourceTag := flag.String("source", ":START_ID", "Tag of the source column")
	targetTag := flag.String("target", ":END_ID", "Tag of the target column")
	flushing := flag.Int64("flushing", 1, "Number of iterations per flush")
	flag.Usage = func() {
		fmt.Printf("Usage: %v [OPTIONS] outputfile csvfile [csvfile...]\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.Parse()

	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(1)
	}

	// get delimiter rune from string
	if len(*delimiterString) == 0 {
		log.Fatalln("Delimiter string should not be empty")
	}
	var delimiter rune
	for _, c := range *delimiterString {
		delimiter = c
		break
	}

	outputFilePath := flag.Arg(0)
	edgeMap := make(map[string]int)
	for i, filePath := range flag.Args()[1:] {
		log.Printf("Evaluating file %v", i)
		fillEdges(&edgeMap, filePath, i, *sourceTag, *targetTag, delimiter)
	}

	writeEdgeList(edgeMap, outputFilePath, *flushing)
}
