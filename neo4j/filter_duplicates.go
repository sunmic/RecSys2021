// Filter duplicate entries from csv files.
// 
// Usage: go run filter_duplicates.go inputfile outputfile source target
// Where `source` and `target` are column names, present in csv header.
// 
// Example - filter relationships:
// > go run filter_duplicates.go inputfile outputfile :START_ID(User-ID) :END_ID(Tweet-ID)
//
// Example - filter nodes (use same column as source and target)
// > go run filter_duplicates.go inputfile outputfile id:ID(User-ID) id:ID(User-ID)

package main

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/cheggaaa/pb"
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

func csvProc(reader io.Reader, writer io.Writer) (*csv.Reader, *csv.Writer) {
	csvReader := csv.NewReader(reader)
	csvReader.Comma = ';'
	csvReader.TrimLeadingSpace = true
	csvReader.ReuseRecord = true
	csvReader.FieldsPerRecord = 0

	csvWriter := csv.NewWriter(writer)
	csvWriter.Comma = ';'
	csvWriter.UseCRLF = false

	return csvReader, csvWriter
}

func fileSize(file *os.File) int64 {
	info, err := file.Stat()
	if err != nil {
		log.Fatal(err)
	}
	return info.Size()
}

func main() {
	if len(os.Args) < 5 {
		fmt.Printf("Usage: %v inputfile outputfile source target\n", os.Args[0])
		os.Exit(1)
	}

	inputFile := os.Args[1]
	outputFile := os.Args[2]
	sourceTag := os.Args[3]
	targetTag := os.Args[4]

	// input file for reading
	fileReader, err := os.Open(inputFile)
	if err != nil {
		log.Fatal(err)
	}

	// output file for writing
	fileWriter, err := os.OpenFile(outputFile, os.O_CREATE|os.O_WRONLY, 0744)

	// create csv reader/writer wrappers
	csvReader, csvWriter := csvProc(fileReader, fileWriter)

	headerRecord, err := csvReader.Read()
	if err != nil {
		log.Fatalf("Could not parse header: %v", err)
	}

	sourceIndex, targetIndex, err := parseHeader(headerRecord, sourceTag, targetTag)
	if err != nil {
		log.Fatal(err)
	}

	csvWriter.Write(headerRecord)

	relationshipSet := make(map[string]bool)
	pbar := pb.New64(fileSize(fileReader))
	pbar.Start()
	var stringBuffer bytes.Buffer
	var totalRead int = 0
	for {
		record, err := csvReader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Println(err)
			continue
		}

		stringBuffer.WriteString(record[sourceIndex])
		stringBuffer.WriteString(record[targetIndex])
		key := stringBuffer.String()
		stringBuffer.Reset()
		totalRead += 1
		if _, ok := relationshipSet[key]; ok {
			continue
		} else {
			relationshipSet[key] = true
			csvWriter.Write(record)
			csvWriter.Flush()
		}

		pos, _ := fileReader.Seek(0, os.SEEK_CUR)
		pbar.Set64(pos)
	}

	pbar.Finish()
	fileReader.Close()
	fileWriter.Close()
	fmt.Printf("Done! Removed %v duplicates. \n", totalRead - len(relationshipSet))
}
