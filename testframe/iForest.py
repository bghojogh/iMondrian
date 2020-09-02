'''
I found that there is some degree of parallelism for scikit-learn iForest code.
However, there is no parallelism for original mondrian forest code
Therefore, I think good idea to write a single thread one myself.
Besides, scikit-learn code is extremely difficult to understand.
Are there any manual to help?
'''
import numpy as np
from sklearn.utils.fixes import euler_gamma


class iForestNode(object):
    """
    defines iForest Node
    variables:
    - min_d         : dimension-wise min of training data in current block
    - max_d         : dimension-wise max of training data in current block
    - left          : id of left child
    - right         : id of right child
    - parent        : id of parent
    - is_leaf       : boolen variable to indicate if current block is leaf
    - depth         : remaining depth
    """

    def __init__(self, depth, parent, number_instance):
        self.number_instance = number_instance
        self.depth = depth
        # real depth is: max_depth - depth; leaf depth = 0 if reach bottom
        self.parent = parent
        self.split_Attr = None
        self.split_val = None
        self.left = None
        self.right = None
        self.is_leaf = True


class iForestTree(object):
    """
    defines a isolation tree
    inputs:
    - data          : N x D numpy array to train this tree
    - hlim          : Depth limit of tree (optional, default: no limit)
    - random_state  : int, seed of rand() (not yet implemented)
    pathLength      : return the path length of ONE single test data sample
        - x         : 1 x D test data sample
        - hlim_path : max path length to trace before c() estimate
                        if (hlim_path > self.max_depth) will be ignored
                        if no input then equals to self.max_depth
        output      : float (estimated) path length
    functions:
    - __init__      : initialize a isolation tree and grow it
    """

    def __init__(self, data=None, hlim=None, random_state=None):
        if data is None:
            return
        if hlim is None:
            hlim = data.shape[0] - 1  # default max depth
        root_node = iForestNode(hlim, None, data.shape[0])
        # self.sample_size = sample_size #can make this automatic
        self.max_depth = hlim
        self.node_list = []  # store struct iForestNode for each tree node
        self.node_list.append(root_node)
        self.non_leaf_nodes = []  # store corresponding index in self.node_list
        self.leaf_nodes = []  # store coresponding index in self.node_list
        # queue/stack of index of self.node_list to grow tree
        self.grow_nodes = [0]
        self.grow(data, random_state)  # grow tree function call

    def grow(self, data, random_state):
        node_data = []  # queue/stack of training data for each tree node
        node_data.append(data)  # order same as self.grow_nodes
        while self.grow_nodes:
            node_index = self.grow_nodes.pop(0)
            curr_data = node_data.pop(0)
            # more than 1 instance in node
            if(self.node_list[node_index].number_instance > 1):
                if(self.node_list[node_index].depth > 0):
                    # have depth remainning
                    # then not a leaf
                    self.node_list[node_index].is_leaf = False
                    feat_id_chosen = np.random.randint(
                        curr_data.shape[1],
                    )  # choose a feature
                    # record chosen feature
                    self.node_list[node_index].split_Attr = feat_id_chosen
                    # min/max of chosen feature
                    split_Attr_max = max(curr_data[:, feat_id_chosen])
                    split_Attr_min = min(curr_data[:, feat_id_chosen])
                    if(split_Attr_max == split_Attr_min):
                        # when randomly selected attr all have same value
                        # check all attr that does not all have same value
                        # check if all data in this node are same
                        each_attr = np.zeros(curr_data.shape[1], dtype=int)
                        for attr in range(curr_data.shape[1]):
                            attr_min = min(curr_data[:, attr])
                            attr_max = max(curr_data[:, attr])
                            if(attr_min != attr_max):
                                each_attr[attr] = 1
                        count_diff_attr = sum(each_attr)
                        if(count_diff_attr == 0):  # all data in node are same
                            self.leaf_nodes.append(node_index)
                            self.node_list[node_index].is_leaf = True
                            continue
                            # then the node is a leaf node
                        # if not all data in this node are same
                        # randomly choose an attr that have a range > 0
                        feat_count_chosen = np.random.randint(count_diff_attr)
                        for attr in range(curr_data.shape[1]):
                            if(each_attr[attr] == 1):
                                if(feat_count_chosen == 0):
                                    feat_id_chosen = attr
                                else:
                                    feat_count_chosen = feat_count_chosen - 1
                        split_Attr_max = max(curr_data[:, feat_id_chosen])
                        split_Attr_min = min(curr_data[:, feat_id_chosen])
                        # min/max of new chosen feature

                    self.node_list[node_index].split_val = np.random.uniform(
                        split_Attr_min, split_Attr_max,
                    )
                    # [min, max) select split_val
                    mask_data_left = (
                        curr_data[:, feat_id_chosen] <=
                        self.node_list[node_index].split_val
                    )
                    # data belonging to left node
                    data_left = curr_data[mask_data_left, :]
                    node_left = iForestNode(
                        (
                            self.node_list[node_index].depth -
                            1
                        ), node_index, data_left.shape[0],
                    )
                    # append data for left node to stack
                    node_data.append(data_left)
                    # append left index to stack
                    self.grow_nodes.append(len(self.node_list))
                    self.node_list[node_index].left = len(
                        self.node_list,
                    )  # set node's left child
                    # append to list of nodes for tree
                    self.node_list.append(node_left)

                    mask_data_right = np.invert(mask_data_left)
                    # data belonging to right node
                    data_right = curr_data[mask_data_right, :]
                    node_right = iForestNode(
                        (
                            self.node_list[node_index].depth -
                            1
                        ), node_index, data_right.shape[0],
                    )
                    node_data.append(data_right)
                    self.grow_nodes.append(len(self.node_list))
                    self.node_list[node_index].right = len(self.node_list)
                    self.node_list.append(node_right)

                    self.non_leaf_nodes.append(node_index)
                else:
                    self.leaf_nodes.append(node_index)
            else:
                self.leaf_nodes.append(node_index)
        return self

    def pathLength(self, x, hlim_path=None):
        # x is a single instance here 1*n array with n attributes
        # x cannot be an array of instances
        if hlim_path is None:
            hlim_path = self.max_depth
        curr_index = 0
        while True:
            curr_depth = self.max_depth - self.node_list[curr_index].depth
            if self.node_list[curr_index].is_leaf:  # is leaf node then output
                return c_val(
                    self.node_list[curr_index].number_instance,
                    curr_depth,
                )
            elif curr_depth > hlim_path:  # reach max_depth then output
                return c_val(
                    self.node_list[curr_index].number_instance,
                    curr_depth,
                )
            attr_index = self.node_list[curr_index].split_Attr
            attr_split = self.node_list[curr_index].split_val
            if x[attr_index] <= attr_split:  # go to left node
                curr_index = self.node_list[curr_index].left
            else:  # go to right node
                curr_index = self.node_list[curr_index].right


class iForest(object):
    """
    defines a isolation forest
    variables:
    - data          : N x D training data to train forest
    - hlim          : depth limit of all tree in forest
                            (optional, default: no limit)
    - forest_size   : number of trees in forest (optional, default 100)
    - contamination : percentage of data that will be anomolies
                            (not yet implemented)
    - random_state  : seed for rand() (not yet implemented)
    predict_item    : return anomly score for a single test sample
        -x          : 1 x D data which anomly score is to be computed
        -hlim       : max path length to trace before c() estimate
                        if (hlim_path > self.max_depth) ignored
                        if no input then equals to self.max_depth
        output      : float anomly score
    decision_function: return anomly score of test sample
        -x          : N x D data with N samples
        -hlim       : max path length to trace before c() estimate
                        if (hlim_path > self.max_depth) ignored
                        if no input then equals to self.max_depth
        output      : numpy float array anomly scores
    functions:
    - __init__      : initialize a isolation forest and grow it
    """

    '''
    sklearn:
        if tree_size = "auto"
        tree_size = min(256, data.shape[0])

    '''

    def __init__(
        self, data=None, hlim=None, forest_size=100,
        contamination=None, random_state=None, with_replacement=False,
        project_flag=False
    ):
        # paper says select data without replacement
        # strict select data without replacement different from scikitlearn
        # contamination not yet implemented
        # random_state not yet implemented
        self.contamination = contamination
        self.with_replacement = with_replacement
        self.hlim = hlim
        self.tree_size = 0
        self.forest_size = int(forest_size)
        self.forest = [None] * forest_size  # list to store iForestTree objects
        self.random_state = random_state
        self.project_flag = project_flag
        if data is None:
            pass
        else:
            temp_tree_size = int(data.shape[0] / forest_size)
            if with_replacement is True:
                self.tree_size = max(256, temp_tree_size)
            else:
                self.tree_size = temp_tree_size
            self.fit(data, None, hlim, forest_size, random_state, project_flag)

    def fit(self, data=None, y=None, hlim=None, forest_size=100, random_state=None,
            project_flag=False):
        if hlim is None:
            hlim = self.hlim
        if(forest_size == 100):
            forest_size = self.forest_size
        else:
            self.forest_size = forest_size
        if random_state is None:
            random_state = self.random_state
        if not project_flag:
            project_flag = self.project_flag
        if(self.tree_size == 0):
            if self.with_replacement is False:
                self.tree_size = int(data.shape[0] / forest_size)
            else:
                self.tree_size = int(max(256, data.shape[0]/forest_size))
        if(self.tree_size * self.forest_size > data.shape[0] and not project_flag):
            # select sample for each tree with replacement
            sample_weight = np.ones(data.shape[0])
            # weight method 1:
            # weight_decrement = 1 / self.forest_size
            # weight method 2:
            weight_decrement = data.shape[0]/(self.forest_size*self.tree_size)
            for i in range(forest_size):
                sum_1_weight = sample_weight / sum(sample_weight)
                data_sample_index = np.random.choice(data.shape[0],
                                                     self.tree_size, replace=False, p=sum_1_weight)
                data_sample = data[data_sample_index, :]
                sample_weight[data_sample_index] -= weight_decrement
                # accompany method 2
                sample_weight[np.where(sample_weight < 0)] = 0
                self.forest[i] = iForestTree(data_sample, hlim, random_state)
        elif(self.tree_size * self.forest_size > data.shape[0] and project_flag):
            sample_weight = np.ones(data.shape[0])
            sample_weight = sample_weight / len(sample_weight)
            for i in range(forest_size):
                data_sample_index = np.random.choice(data.shape[0],
                                                     self.tree_size, replace=False, p=sample_weight)
                data_sample = data[data_sample_index, :]
                self.forest[i] = iForestTree(data_sample, hlim, random_state)
        else:
            # select sample for each tree without replacement
            # strictly as paper specified
            curr_tree_start = 0
            curr_tree_end = self.tree_size
            for i in range(forest_size):
                data_sample = data[curr_tree_start:curr_tree_end, :]
                self.forest[i] = iForestTree(data_sample, hlim, random_state)
                curr_tree_start = curr_tree_end
                curr_tree_end = curr_tree_end + self.tree_size
        # for now just take slices from original data
        # assuming original data is shuffled
        return self

    def predict_item(self, x, hlim=None):  # currently return score
        sum_score = 0.  # add method to output prediction result
        for i in range(self.forest_size):
            curr_score = self.forest[i].pathLength(x, hlim)
            sum_score = sum_score + curr_score
        E_score = sum_score / self.forest_size
        # use curr_depth = 0 to avoid calculate it
        overall_c = c_val(self.tree_size, 0)
        return 2 ** (-E_score / overall_c)

    def decision_function(self, x, hlim=None):  # currently return score
        x = np.array(x)
        test_samples = x.shape[0]  # add method to output prediction result
        test_score = np.zeros(test_samples)
        for i in range(test_samples):
            test_score[i] = self.predict_item(x[i, :], hlim)
        return test_score

    def predict(self, x, hlim=None):
        x = np.array(x)
        ans = np.zeros(x.shape[0])
        return ans


def c_val(sub_size, curr_depth):  # c(sub_size) equation in paper
    if sub_size == 1:  # has one instance
        return float(curr_depth)  # c() = 0
    elif sub_size == 2:  # has two instances
        return float(curr_depth + 1.)
    else:  # has more instances
        c = np.log(sub_size - 1.) + euler_gamma - (sub_size - 1.) / sub_size
        c = 2 * c
        return curr_depth + c


def main():
    import matplotlib.pyplot as plt
    # from matplotlib.colors import ListedColormap
    rng = np.random.RandomState(42)

    # Generate train data
    X = 0.3 * rng.randn(1000, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * rng.randn(2, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(2, 2))

    # fit the model
    tree = iForest(data=X_train, forest_size=20, with_replacement=True)
    # test_score = tree.predict(X_test)
    # anom_score = tree.predict(X_outliers)
    # print(test_score)
    # print(anom_score)
    # print tree.pathLength(X_outliers[1,:])

    h = .08  # step size in the mesh

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = tree.decision_function(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z)

    # Plot also the training points
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                     s=20, edgecolor='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend([b1, b2, c],
               ['training observations',
                'new regular observations', 'new abnormal observations'],
               loc='upper left')
    plt.show()
    '''

    #y_pred_train = clf.predict(X_train)
    y_pred_test = tree.predict(X_test)
    y_pred_outliers = tree.predict(X_outliers)

    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                     s=20, edgecolor='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([b1, b2, c],
               ["training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left")
    plt.show()
    '''


if __name__ == '__main__':
    main()
