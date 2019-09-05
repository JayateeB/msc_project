from mrjob.job import MRJob
import json


class MREdgeListToAdjacencyList(MRJob):

    def mapper(self, _, line):
        fields = line.split(",")
        source = fields[0]
        target = fields[1]
        yield source, target

    def reducer(self, key, values):
        yield key, list(values)


if __name__ == '__main__':
    MREdgeListToAdjacencyList.run()
