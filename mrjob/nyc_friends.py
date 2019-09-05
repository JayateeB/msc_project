from mrjob.job import MRJob

infected_users = set(int(line.strip()) for line in open('/Users/syamantak/JayateeB/dissertation/input/nyc_infected_9_hrs.csv'))


class MRNYCFriends(MRJob):

    def mapper(self, _, line):
        data = line.split(',')
        if int(data[1]) in infected_users:
            yield int(data[0]), int(data[1])

    def reducer(self, user, followers):
        yield user, list(followers)


if __name__ == '__main__':
    MRNYCFriends.run()