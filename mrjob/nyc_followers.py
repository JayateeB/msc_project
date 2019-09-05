from mrjob.job import MRJob

class nyc_followers(MRJob):

    def mapper(self, _, line):
        data = line.split(',')
        yield data[1],data[0]

    def reducer(self, user, followers):
        yield user,list(followers)


if __name__ == '__main__':
    nyc_followers.run()