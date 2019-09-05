from mrjob.job import MRJob
import json


class MRFriendsToFollowers(MRJob):

    def mapper(self, _, line):
        fields = line.split("\t")
        source = fields[0]
        friends = json.loads(fields[1])
        if len(friends) > 0:
            for t in friends:
                yield t, source

    def reducer(self, key, values):
        yield key, list(values)


if __name__ == '__main__':
    MRFriendsToFollowers.run()