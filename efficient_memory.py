import torch





class Filter:
    def __init__(self):
        pass

    def select(self, features: list, select_every: int = 5):
        """
        Args:
            features (list of torch.Tensor): shape (b, t, d)  eg. [(2700, 768), (2800, 768), (2900, 768), (3000, 768)]
            select_every (int): select 1 frame every select_every frames
        Returns:
            shape (b, 0.2 * t, d), shape (b, 0.8 * t, d)
        """
        selected_features = []
        unselect_features = []
        for i in range(len(features)):
            feature = features[i]
            t, d = feature.shape
            # Select one frame and leave out select_every - 1 frames
            selected_features.append(feature[::select_every, :])
            unselect_features.append(feature[[i for i in range(t) if i % select_every != 0], :])

        return selected_features, unselect_features



class EfficientMemory:
    def __init__(self):
        self.init = True  # need to init

    def init_memory(self, features: torch.Tensor):
        self.memory = features  # NOTE: 可能有个问题，就是这样做假定了这个features是互不相关的
        self.init = False
        self.refer_video = [None] * len(features)  # lenth = batch size
        print('init memory.shape: ', (len(features), features[0].shape))


    def update(self, new_features: torch.Tensor, similarity_threshold: float = 0.7):
        """
        Update memory with new features based on similarity.
        """
        for batch in range(len(new_features)):
            feature = new_features[batch]
            # Cosine similarity between the feature and each feature in memory
            similarities = torch.nn.functional.cosine_similarity(self.memory, feature.unsqueeze(0), dim=1)
            # Check if all similarities are below the threshold
            if torch.all(similarities < similarity_threshold):
                self.memory = torch.cat((self.memory, feature.unsqueeze(0)), dim=0)  # shape (b+1, t, d)
                self.refer_video[batch] = self.memory.size(0) - 1
            else:
                # find the most similar feature
                most_similar_index = torch.argmax(similarities)
                self.refer_video[batch] = most_similar_index


    def insert(self, features: torch.Tensor, insert_num: int = 100, similarity_threshold: float = 0.7):
        """
        Insert selected memory into the features
        Args:
            features: shape (4, 540, 768)
            insert_num: int = 100
        Returns:
            features: shape (4, 540 + 100, 768)
        """
        for batch in range(len(features)):
            feature = features[batch]  # shape (540, 768)
            refer_index = self.refer_video[batch]  # find in the most similar feature in memory
            # TODO: debug self.memory.shape is not (2160, 768)
            memory_feature = self.memory[refer_index]  # shape (2160, 768)
            print('memory_feature.shape: ', memory_feature.shape)

            cos_similarity_matrix = torch.nn.functional.\
                cosine_similarity(feature, memory_feature, dim=1)  # shape (540, 2160)
            
            for i in range(cos_similarity_matrix.size(0)):  # 540
                cos_similarity_vector = cos_similarity_matrix[i]  # shape (2160)
                similar_vector = []  # 540
                similar_index = []  # 540 each (0 - 2159)
                insert_index = []  # 540 each (1 - 540)  index to insert
                for j in range(cos_similarity_matrix.size(0)):  # 540
                    # find the most unsimilar one in j:j+4
                    most_unsimilar_index = torch.argmin(cos_similarity_vector[j:j+4])
                    similar_vector.append(cos_similarity_vector[most_unsimilar_index])
                    similar_index.append(most_unsimilar_index)
                    insert_index.append(j+1)

                # sort the similar_index by similar_vector
                sorted_index = sorted(range(len(similar_vector)), key=lambda x: similar_vector[x])
                final_index = sorted_index[:insert_num]

                # insert the features into the features without deleting original data
                for idx in final_index:
                    features[batch] = torch.cat((features[batch][:insert_index[idx]], 
                                                 memory_feature[similar_index[idx]].unsqueeze(0), 
                                                 features[batch][insert_index[idx]:]), dim=0)

        return features
















