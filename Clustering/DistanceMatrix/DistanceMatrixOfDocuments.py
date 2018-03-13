from sklearn.metrics.pairwise import euclidean_distances

class EuclideanDistanceForDocuments():

    def euclidean_distance_matrix(self,X, y, exemplars):
        centeriod_and_data_radius = dict()
        for index, value in enumerate(y):
            print("THE x VALUE BY INDEX", X[index])
            print("THE x VALUE BY exemplars", exemplars[value])
            radius = euclidean_distances([X[index]], [exemplars[value]])[0, 0]
            value = int(value)
            value = 'clusters ' + str(value)
            if value not in centeriod_and_data_radius.keys():
                data_radius_list = list()
                data_radius_list.append(radius)
                centeriod_and_data_radius[value] = data_radius_list
            else:
                data_radius_list = centeriod_and_data_radius[value]
                data_radius_list.append(radius)
                centeriod_and_data_radius[value] = data_radius_list


        return centeriod_and_data_radius